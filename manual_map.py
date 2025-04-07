import itertools

# Define currencies (B, C, D, A) and exchange rates
currencies = ['B', 'C', 'D', 'A']
rates = [
    # From B (row 0)
    [1,    1.45, 0.52, 0.72],  # To B, C, D, A
    # From C (row 1)
    [0.7,  1,    0.31, 0.48],  
    # From D (row 2)
    [1.95, 3.1,  1,    1.49],  
    # From A (row 3)
    [1.34, 1.98, 0.64, 1]      
]

start_end = 3  # Must start/end with A (SeaShells)
max_trades = 5

# Collect all possible paths and their profits
all_strategies = []

for num_trades in range(1, max_trades + 1):
    intermediates = num_trades - 1
    for path in itertools.product(range(4), repeat=intermediates):
        full_path = [start_end] + list(path) + [start_end]
        product = 1.0
        for i in range(len(full_path) - 1):
            from_curr = full_path[i]
            to_curr = full_path[i + 1]
            product *= rates[from_curr][to_curr]
        all_strategies.append((product, full_path))

# Sort strategies by descending profit and remove duplicates
all_strategies.sort(reverse=True, key=lambda x: x[0])
seen = set()
top_10 = []
for profit, path in all_strategies:
    path_key = tuple(path)
    if path_key not in seen:
        seen.add(path_key)
        top_10.append((profit, path))
    if len(top_10) >= 10:
        break

# Display results
print("Top 10 Strategies Ranked by Profit:\n")
for rank, (profit, path) in enumerate(top_10, 1):
    path_named = [currencies[i] for i in path]
    trades = len(path) - 1  # Number of trades = path length -1
    profit_pct = (profit - 1) * 100
    print(f"#{rank}:")
    print(f"Path ({trades} trades): {' → '.join(path_named)}")
    print(f"Profit: {profit_pct:.2f}% (Multiplier: {profit:.4f})\n")