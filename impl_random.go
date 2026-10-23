package num

import (
	"fmt"
	"math"
)

// —— 随机数(Step 8c):逐位复刻 numpy 的 RandomState(MT19937)路径 ——
//
// 兼容目标:numpy 1.x/2.x 的 `np.random.RandomState(seed)`(legacy 接口)。
// 复刻的算法细节:
// - MT19937 状态机 + `init_genrand` 播种(标量种子,取低 32 位);
// - `random_sample` = ((a>>5)·2^26 + (b>>6)) / 2^53(每次取两个 32 位字);
// - `rk_gauss` 极坐标法(**带缓存 spare**,故 randn 必须连续调用才与 numpy 对齐);
// - `rk_interval` 掩码拒绝采样(用于 randint / permutation);
// - permutation/shuffle 用 Fisher-Yates(从尾部往前)。
//
// 这与 numpy 的 Generator(PCG64)不同;后者不提供逐位兼容。

// MT19937 状态。
type mt19937 struct {
	mt       [624]uint32
	idx      int
	hasGauss bool
	gauss    float64
}

func (m *mt19937) seed(seed uint32) {
	m.mt[0] = seed
	for i := 1; i < 624; i++ {
		m.mt[i] = 1812433253*(m.mt[i-1]^(m.mt[i-1]>>30)) + uint32(i)
	}
	m.idx = 624
	m.hasGauss = false
	m.gauss = 0
}

func (m *mt19937) genrand() uint32 {
	if m.idx >= 624 {
		for i := 0; i < 624; i++ {
			y := (m.mt[i] & 0x80000000) | (m.mt[(i+1)%624] & 0x7fffffff)
			next := m.mt[(i+397)%624] ^ (y >> 1)
			if y&1 != 0 {
				next ^= 0x9908b0df
			}
			m.mt[i] = next
		}
		m.idx = 0
	}
	y := m.mt[m.idx]
	m.idx++
	y ^= y >> 11
	y ^= (y << 7) & 0x9d2c5680
	y ^= (y << 15) & 0xefc60000
	y ^= y >> 18
	return y
}

// double ≡ rk_double:53 位精度。
func (m *mt19937) double() float64 {
	a := int64(m.genrand() >> 5)
	b := int64(m.genrand() >> 6)
	return (float64(a)*67108864.0 + float64(b)) / 9007199254740992.0
}

// gaussSample ≡ rk_gauss(极坐标法,带 spare 缓存)。
func (m *mt19937) gaussSample() float64 {
	if m.hasGauss {
		m.hasGauss = false
		return m.gauss
	}
	var x1, x2, r2 float64
	for {
		x1 = 2.0*m.double() - 1.0
		x2 = 2.0*m.double() - 1.0
		r2 = x1*x1 + x2*x2
		if r2 < 1.0 && r2 != 0.0 {
			break
		}
	}
	f := math.Sqrt(-2.0 * math.Log(r2) / r2)
	m.gauss = f * x1
	m.hasGauss = true
	return f * x2
}

// interval ≡ rk_interval:均匀取 [0, max] 的整数(掩码拒绝采样)。
func (m *mt19937) interval(max uint32) uint32 {
	if max == 0 {
		return 0
	}
	mask := max
	mask |= mask >> 1
	mask |= mask >> 2
	mask |= mask >> 4
	mask |= mask >> 8
	mask |= mask >> 16
	for {
		v := m.genrand() & mask
		if v <= max {
			return v
		}
	}
}

// interval64 64 位区间(用于大范围 randint)。
func (m *mt19937) interval64(max uint64) uint64 {
	if max == 0 {
		return 0
	}
	mask := max
	mask |= mask >> 1
	mask |= mask >> 2
	mask |= mask >> 4
	mask |= mask >> 8
	mask |= mask >> 16
	mask |= mask >> 32
	for {
		v := (uint64(m.genrand()) << 32) | uint64(m.genrand())
		v &= mask
		if v <= max {
			return v
		}
	}
}

// RandomState ≡ np.random.RandomState(legacy MT19937 生成器)。
type RandomState struct {
	mt mt19937
}

// NewRandomState ≡ np.random.RandomState(seed)。
func NewRandomState(seed int64) *RandomState {
	r := &RandomState{}
	r.Seed(seed)
	return r
}

// Seed ≡ ndarray.random.seed:重置状态(取种子低 32 位,numpy 同)。
func (r *RandomState) Seed(seed int64) { r.mt.seed(uint32(uint64(seed))) }

// RandomSample ≡ np.random.rand / random_sample:U[0,1) 的 size 个样本。
func (r *RandomState) RandomSample(size int) (*Array, error) {
	out, err := New([]int{size}, DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(out))
	for i := range w {
		w[i] = r.mt.double()
	}
	return out, nil
}

// StandardNormal ≡ np.random.randn:标准正态(极坐标法,连续调用才与 numpy 逐位一致)。
func (r *RandomState) StandardNormal(size int) (*Array, error) {
	out, err := New([]int{size}, DFloat64)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(out))
	for i := range w {
		w[i] = r.mt.gaussSample()
	}
	return out, nil
}

// Uniform ≡ np.random.uniform(low, high, size)。
func (r *RandomState) Uniform(low, high float64, size int) (*Array, error) {
	out, err := r.RandomSample(size)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(out))
	for i := range w {
		w[i] = low + (high-low)*w[i]
	}
	return out, nil
}

// Normal ≡ np.random.normal(loc, scale, size)。
func (r *RandomState) Normal(loc, scale float64, size int) (*Array, error) {
	out, err := r.StandardNormal(size)
	if err != nil {
		return nil, err
	}
	w := win[float64](engineWindow(out))
	for i := range w {
		w[i] = loc + scale*w[i]
	}
	return out, nil
}

// RandInt ≡ np.random.randint(low, high, size):区间 [low, high),结果 int64。
func (r *RandomState) RandInt(low, high int64, size int) (*Array, error) {
	if high <= low {
		return nil, fmt.Errorf("num: randint: high(%d) 必须大于 low(%d): %w", high, low, ErrValue)
	}
	out, err := New([]int{size}, DInt64)
	if err != nil {
		return nil, err
	}
	w := win[int64](engineWindow(out))
	width := uint64(high - low) // max = width-1
	max := width - 1
	for i := range w {
		if width <= 1<<32 {
			w[i] = low + int64(r.mt.interval(uint32(max)))
		} else {
			w[i] = low + int64(r.mt.interval64(max))
		}
	}
	return out, nil
}

// Permutation ≡ np.random.permutation(n):返回 0..n-1 的随机排列(Fisher-Yates)。
func (r *RandomState) Permutation(n int) (*Array, error) {
	out, err := New([]int{n}, DInt64)
	if err != nil {
		return nil, err
	}
	w := win[int64](engineWindow(out))
	for i := range w {
		w[i] = int64(i)
	}
	// 与 numpy 同:从尾部往前,每次在 [0, i] 取一个位置交换
	for i := n - 1; i > 0; i-- {
		j := int(r.mt.interval(uint32(i)))
		w[i], w[j] = w[j], w[i]
	}
	return out, nil
}

// Shuffle ≡ np.random.shuffle(a):沿第 0 轴原地打乱(其余维度作为整体搬移)。
func (r *RandomState) Shuffle(a *Array) error {
	if a.NDim() == 0 {
		return fmt.Errorf("num: shuffle: 0-d 输入无法打乱: %w", ErrShape)
	}
	n := a.Shape()[0]
	es := a.Dtype().Itemsize
	buf := engineWindow(a)
	rowLen := es
	for _, d := range a.Shape()[1:] {
		rowLen *= d
	}
	swap := func(i, j int) {
		if i == j {
			return
		}
		a0 := buf[i*rowLen : (i+1)*rowLen]
		b0 := buf[j*rowLen : (j+1)*rowLen]
		tmp := make([]byte, rowLen)
		copy(tmp, a0)
		copy(a0, b0)
		copy(b0, tmp)
	}
	for i := n - 1; i > 0; i-- {
		j := int(r.mt.interval(uint32(i)))
		swap(i, j)
	}
	return nil
}
