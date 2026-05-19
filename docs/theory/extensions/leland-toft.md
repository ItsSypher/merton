# Leland-Toft endogenous default

Merton, Black-Cox, and CreditGrades all treat the default boundary as
**exogenous**: someone (analyst, regulator, indenture writer) tells the
model when default occurs. Leland (1994) and Leland-Toft (1996) flip
this on its head — equity holders **choose** the default boundary
``V_B^*`` to maximise their own claim, weighing the option value of
continuing to pay coupons against the loss of the firm's tax shield.

## Setup

- Assets follow GBM under the risk-neutral measure with dividend yield
  ``\delta``: ``dV/V = (r - \delta)\,dt + \sigma_A\,dW``.
- Debt pays a continuous coupon ``C``.
- Tax rate ``\tau`` (coupon payments shielded from tax).
- Bankruptcy cost ``\alpha`` (debt holders recover ``(1 - \alpha) V_B``).

## Optimal default boundary

Let ``x`` be the positive root of the characteristic quadratic
``\tfrac{1}{2}\sigma_A^2 y^2 + (r - \delta - \sigma_A^2/2)y - r = 0``:

$$x = \frac{\sqrt{(r - \delta - \sigma_A^2/2)^2 + 2r\sigma_A^2}
       - (r - \delta - \sigma_A^2/2)}{\sigma_A^2}.$$

The smooth-pasting condition yields

$$V_B^* = \frac{(1 - \tau)\,C\,x}{r\,(1 + x)}.$$

## Default probability

For ``V > V_B``:

$$\mathrm{PD} = \left(\frac{V_B}{V}\right)^x \in (0, 1].$$

(Below the boundary, ``\mathrm{PD} = 1`` — the firm is already
in default.)

## Equity and debt values

$$E(V) = V - \frac{(1 - \tau)C}{r}\,\bigl(1 - \mathrm{PD}\bigr)
         - (1 - \alpha)\,V_B\,\mathrm{PD},$$

$$D(V) = \frac{C}{r}\,\bigl(1 - \mathrm{PD}\bigr) + (1 - \alpha)\,V_B\,\mathrm{PD}.$$

The Modigliani-Miller decomposition ``E + D = V + \text{tax shield} -
\text{bankruptcy cost}`` is preserved.

## Example

```{code-block} python
from merton import Firm
from merton.extensions import LelandToftModel

firm = Firm(equity=30, debt_short=30, debt_long=40, equity_vol=0.50,
            rf=0.04, horizon=1.0)
result = LelandToftModel(
    coupon=5.0, tax_rate=0.25, bankruptcy_cost=0.30,
).fit(firm)
print(result.summary())
print("V_B*:", result.default_point)
print("PD :", result.pd)
```

## When to prefer Leland-Toft

- **Capital-structure analytics.** If you're varying the coupon or tax
  treatment, the endogenous boundary is exactly what you want.
- **LBO and recap modeling.** The optimal-stopping formulation captures
  how new debt issuance changes the default decision.
- **Bond pricing with embedded covenants.** Pairing Leland-Toft with the
  Vasicek single-factor portfolio engine gives a self-consistent
  framework for both pricing and capital.

For "what's this firm's one-year PD?" Merton or KMV remain the simpler
choices.
