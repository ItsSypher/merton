# Black-Cox first-passage model

Merton (1974) treats default as a single check at the debt-maturity horizon
``T``: the firm fails only if ``A_T < D``. Black & Cox (1976) ask "what if
the lender can call the loan whenever ``A_t`` falls below a covenant
barrier ``K(t)``"? The model becomes a *first-passage time* problem.

## Default formula

Under risk-neutral dynamics ``dA = (r-q)A dt + σ_A A dW``, with constant
barrier ``K``:

$$
\mathrm{PD}_{BC}
= \Phi\!\left(\frac{-a - \nu T}{\sigma_A \sqrt{T}}\right)
  + \exp\!\left(-\frac{2 \nu a}{\sigma_A^2}\right)\,
    \Phi\!\left(\frac{-a + \nu T}{\sigma_A \sqrt{T}}\right),
$$

where ``a = log(A_0/K) > 0`` and ``ν = r - q - σ_A^2/2``.

When the barrier grows as ``K(t) = K_T · e^{-γ(T-t)}``, the change of
variables ``Y_t = log(A_t / K(t))`` keeps the formula intact with
``ν → ν - γ``.

## Practical comparison

Black-Cox PD ≥ Merton PD for the same firm: first-passage adds default
opportunities, so the probability of *touching* the barrier at any time
in `[0, T]` is at least the probability of *ending* below it at `T`.

```{code-block} python
from merton import Firm, fit
from merton.extensions import BlackCoxModel

firm = Firm(equity=50, debt_short=30, debt_long=50, equity_vol=0.5, rf=0.04, horizon=1.0)
m = fit(firm, method="jmr_iterative")
bc = BlackCoxModel().fit(firm)
print(f"Merton PD:   {m.pd:.4f}")
print(f"Black-Cox PD:{bc.pd:.4f}")
```

## When to prefer Black-Cox

- Public companies with strict covenants where breach triggers immediate
  default (the covenant level is the natural barrier).
- Term loans with maintenance covenants.
- Sovereign / project-finance contexts with credit-event triggers.

For straightforward "default at maturity" instruments (zero-coupon bonds),
Merton is the right baseline.
