import gzip

samples = {}
cur = None

with gzip.open('GSE226260_family.soft.gz', 'rt', encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if line.startswith('^SAMPLE'):
            cur = line.split('=')[1].strip()
            samples[cur] = []
        elif cur and 'characteristics_ch1' in line:
            val = line.split('=', 1)[1].strip()
            samples[cur].append(val)

print(f"Toplam GSM ornegi: {len(samples)}")

# Ilk 3 ornegi goster
for i, (gsm, chars) in enumerate(samples.items()):
    if i >= 5: break
    print(f"\n--- {gsm} ---")
    for c in chars:
        print(f"  {c}")

# Tum benzersiz key'leri bul
all_keys = set()
for chars in samples.values():
    for c in chars:
        if ':' in c:
            all_keys.add(c.split(':')[0].strip())

print(f"\nTum benzersiz ozellik isimleri ({len(all_keys)}):")
for k in sorted(all_keys):
    print(f"  {k}")

# pasc status dagilimi
pasc_counts = {}
for gsm, chars in samples.items():
    for c in chars:
        if 'pasc status' in c.lower():
            val = c.split(':')[1].strip()
            pasc_counts[val] = pasc_counts.get(val, 0) + 1

print(f"\nPASC Status dagilimi: {pasc_counts}")

# infection status veya disease state var mi?
infection_counts = {}
for gsm, chars in samples.items():
    for c in chars:
        if 'infection' in c.lower() or 'disease' in c.lower() or 'group' in c.lower() or 'condition' in c.lower():
            infection_counts[c] = infection_counts.get(c, 0) + 1

print(f"\nInfection/Disease/Group alanlari:")
for k, v in sorted(infection_counts.items(), key=lambda x: -x[1]):
    print(f"  {v}x -> {k}")
