from scipy.optimize import linprog
import numpy as np
import pandas as pd

def optimization_selection(df):
    n = len(df)
    products = ['mf', 'cc', 'cl']

    # Prepare expected revenue vector
    exp_revs = np.zeros(3 * n)
    for i, p in enumerate(products):
        exp_revs[i*n:(i+1)*n] = df[f'prob_{p}'] * df[f'Revenue_{p.upper()}']

    # Objective: maximize expected revenue
    c = -exp_revs

    # Constraints:

    # Each customer gets at most one offer:
    A_ub = np.zeros((n + 1, 3*n))
    b_ub = np.zeros(n + 1)

    for i in range(n):
        # sum of products assigned to customer i <= 1
        A_ub[i, i] = 1          # mf variable
        A_ub[i, i + n] = 1      # cc variable
        A_ub[i, i + 2*n] = 1    # cl variable
        b_ub[i] = 1

    # Total offers ≤ 15% of all customers
    A_ub[n, :] = 1
    b_ub[n] = 0.15 * n

    # Bounds: variables between 0 and 1 (relaxed)
    bounds = [(0, 1)] * (3 * n)

    # Solve LP
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')

    if not res.success:
        raise RuntimeError("Optimization failed:", res.message)

    x = res.x

    # Post-process results: pick product with max fractional value for each customer
    offers = []
    for i in range(n):
        vals = [x[i], x[i + n], x[i + 2*n]]
        max_val = max(vals)
        if max_val > 0:
            prod_idx = vals.index(max_val)
            prod = products[prod_idx].upper()
            expected_revenue = exp_revs[prod_idx * n + i]
            offers.append({
                'Client': df.iloc[i]['Client'],        # Use iloc to avoid KeyError
                'Product': prod,
                'Expected_Revenue': expected_revenue
            })

    # Enforce offer cap by sorting and selecting top 15%
    offers_df = pd.DataFrame(offers)
    offers_df = offers_df.sort_values('Expected_Revenue', ascending=False).head(int(0.15 * n)).reset_index(drop=True)

    return offers_df

