"""Fix docs files to be cp1252 compatible by replacing Unicode chars with ASCII equivalents."""

REPLACEMENTS = {
    # Box drawing characters
    '\u250c': '+', '\u2510': '+', '\u2514': '+', '\u2518': '+',
    '\u251c': '+', '\u2524': '+', '\u252c': '+', '\u2534': '+', '\u253c': '+',
    '\u2500': '-', '\u2502': '|', '\u2503': '|',
    '\u2550': '=', '\u2551': '|',
    '\u2554': '+', '\u2557': '+', '\u255a': '+', '\u255d': '+',
    '\u2560': '+', '\u2563': '+', '\u2566': '+', '\u2569': '+', '\u256c': '+',
    # Arrows
    '\u2192': '->', '\u2190': '<-', '\u2193': 'v', '\u2191': '^',
    '\u25b6': '>', '\u25c0': '<',
    '\u25b7': '>', '\u25c1': '<',
    # Dashes
    '\u2014': '--', '\u2013': '-',
    # Checkmarks / symbols
    '\u2705': '[OK]', '\u274c': '[FAIL]', '\u26a0': '[WARNING]',
    '\u2714': '[OK]', '\u2716': '[FAIL]',
    # Smart quotes
    '\u2019': "'", '\u2018': "'", '\u201c': '"', '\u201d': '"',
    # Ellipsis
    '\u2026': '...',
    # Other box drawing
    '\u2508': '-', '\u2509': '-', '\u250a': '|', '\u250b': '|',
    '\u254c': '-', '\u254d': '|',
    # Triangles
    '\u25b2': '^', '\u25bc': 'v',
    # Bullet points (special)
    '\u2022': '*', '\u25cf': '*',
    # Other common non-cp1252
    '\u2248': '~=', '\u2260': '!=', '\u2264': '<=', '\u2265': '>=',
    '\u00b1': '+/-', '\u00d7': 'x', '\u00f7': '/',
    '\u00b2': '2', '\u00b3': '3',
    '\u03b1': 'alpha', '\u03b2': 'beta', '\u03b3': 'gamma',
    '\u03b4': 'delta', '\u03b5': 'epsilon',
}

for fname in ['docs/architecture.md', 'docs/ax.md']:
    with open(fname, 'r', encoding='utf-8') as f:
        content = f.read()

    for char, replacement in REPLACEMENTS.items():
        content = content.replace(char, replacement)

    # Check for remaining problematic chars
    remaining = []
    for i, c in enumerate(content):
        try:
            c.encode('cp1252')
        except UnicodeEncodeError:
            remaining.append((i, ord(c), repr(c)))

    if remaining:
        print(f'{fname}: Still has {len(remaining)} problematic chars:')
        for pos, code, ch in remaining[:10]:
            print(f'  pos={pos} U+{code:04X} {ch}')
        # Replace remaining with '?'
        result = []
        for c in content:
            try:
                c.encode('cp1252')
                result.append(c)
            except UnicodeEncodeError:
                result.append('?')
        content = ''.join(result)
    else:
        print(f'{fname}: All chars are cp1252 compatible')

    with open(fname, 'w', encoding='utf-8') as f:
        f.write(content)

print('Done fixing docs encoding.')
