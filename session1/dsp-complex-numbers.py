# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.4
#   kernelspec:
#     display_name: .venv (3.14.4)
#     language: python
#     name: python3
# ---

# %%
# display these complex numbers in the form a + ib and their respective polar forms using Euler's formula

import cmath
import matplotlib.pyplot as plt

numbers = [
    ("c1", complex(0, -1)),
    ("c2", complex(1, 1)),
    ("c3", complex(0, 0)),
    ("c4", complex(2, 1)),
]

for name, z in numbers:
    a, b = z.real, z.imag
    r = abs(z)
    theta = cmath.phase(z)

    a_str = f"{a:.0f}"
    b_str = f"{abs(b):.0f}"
    sign = "+" if b >= 0 else "-"

    print(f"{name} = {a_str} {sign} {b_str}i")
    if r == 0:
        print("  Euler form: 0")
    else:
        print(f"  Euler form: {r:.2f} * e^(i {theta:.2f})")
    print()

xs = [z.real for _, z in numbers]
ys = [z.imag for _, z in numbers]
labels = [name for name, _ in numbers]

fig, ax = plt.subplots()
ax.axhline(0, color='gray', linewidth=1)
ax.axvline(0, color='gray', linewidth=1)
ax.scatter(xs, ys, color='blue')
for name, x, y in zip(labels, xs, ys):
    ax.text(x + 0.08, y + 0.08, name, fontsize=10)

max_range = max(max(abs(x) for x in xs), max(abs(y) for y in ys), 1)
ax.set_xlim(-max_range - 1, max_range + 1)
ax.set_ylim(-max_range - 1, max_range + 1)
ax.set_xlabel('Real')
ax.set_ylabel('Imaginary')
ax.set_title('Complex numbers on the complex plane')
ax.set_aspect('equal', adjustable='box')
ax.grid(True, linestyle='--', alpha=0.5)
plt.show()
