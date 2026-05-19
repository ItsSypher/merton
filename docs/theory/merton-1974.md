# The Merton 1974 model

Merton (1974) treats firm equity as a European call option on the firm's
underlying assets.

## Setup

A firm has total asset value $A_t$ that follows geometric Brownian motion
under the risk-neutral measure:

$$
\mathrm{d}A_t = (r - q)\, A_t\, \mathrm{d}t + \sigma_A A_t\, \mathrm{d}W_t.
$$

At maturity $T$ the firm owes a single liability $D$.

- If $A_T \geq D$, equity holders receive $A_T - D$ and bondholders are paid in
  full.
- If $A_T < D$, equity holders walk away (limited liability); the firm
  defaults; bondholders receive $A_T$.

Equity therefore has the payoff $E_T = \max(A_T - D, 0)$ — a European call.

## Equity value via Black-Scholes-Merton

$$
E_0 = A_0\, e^{-qT} \Phi(d_1) - D\, e^{-rT} \Phi(d_2)
$$

with

$$
d_1 = \frac{\ln(A_0 / D) + (r - q + \tfrac{1}{2}\sigma_A^2)\, T}{\sigma_A \sqrt{T}},
\qquad d_2 = d_1 - \sigma_A \sqrt{T}.
$$

## Distance to default and PD

Define the **distance to default** as $d_2$:

$$
\mathrm{DD} = d_2
= \frac{\ln(A_0 / D) + (r - q - \tfrac{1}{2}\sigma_A^2)\, T}{\sigma_A \sqrt{T}}.
$$

The risk-neutral **probability of default** is

$$
\mathrm{PD}_Q = \Phi(-d_2) = \Phi(-\mathrm{DD}).
$$

Converting to the physical measure uses the asset-class Sharpe ratio
$\lambda = (\mu - r)/\sigma$:

$$
\mathrm{DD}_P = \mathrm{DD}_Q + \lambda \sigma_A \sqrt{T},
\qquad \mathrm{PD}_P = \Phi(-\mathrm{DD}_P).
$$

## Implied credit spread

For a horizon $T$ and loss-given-default $\mathrm{LGD}$, the implied
continuous-compounding spread is

$$
s = -\frac{1}{T}\ln\!\bigl(1 - \mathrm{PD}\cdot\mathrm{LGD}\bigr).
$$

## Inferring $A_0$ and $\sigma_A$

Both are unobservable. The classic Jones-Mason-Rosenfeld system uses the
observed equity value $E_0$ and equity volatility $\sigma_E$ together with
Itô's lemma to solve

$$
E_0 = A_0 e^{-qT} \Phi(d_1) - D e^{-rT} \Phi(d_2),
\qquad \sigma_E E_0 = e^{-qT}\Phi(d_1)\sigma_A A_0,
$$

for the two unknowns $A_0$ and $\sigma_A$. Vassalou-Xing extend the procedure
to use the full equity time series under maximum likelihood; the snapshot
case collapses back to the JMR two-equation system.

## Limitations

- A single debt maturity is assumed (Geske handles compound debt structures).
- The default barrier is reached only at $T$ (Black-Cox allows touch-default
  at any $t \leq T$).
- The asset process is pure Brownian (Zhou's jump-diffusion adds jumps).
- Interest rates are constant (Longstaff-Schwartz adds stochastic rates).
- All values are assumed to be lognormal (CreditGrades, Leland-Toft relax this).

All of these refinements ship under {mod}`merton.extensions`.
