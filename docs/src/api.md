# API 参考

## 类型与构造

```@docs
FermionicGaussianState
BosonicGaussianState
fermionic_bdg_matrix
bosonic_bdg_matrix
```

## 访问器

```@docs
nmodes
```

协方差不经导出函数访问：任何熵计算后可直接读字段 `s.Γ`
（首次需要时可用非导出的 `GaussianStates.covariance(s)` 触发计算并缓存）。

## 熵

```@docs
osee
entropy
```

## 稠密构造（调试用）

```@docs
densitymatrix
```
