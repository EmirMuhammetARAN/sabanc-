import requests

genes = [
    'ENSG00000279168', 'ENSG00000171517', 'ENSG00000147570',
    'ENSG00000204850', 'ENSG00000172987', 'ENSG00000258810',
    'ENSG00000286423', 'ENSG00000275243', 'ENSG00000227036',
    'ENSG00000147206'
]

server = 'https://rest.ensembl.org'
ext = '/lookup/id'
headers = { 'Content-Type' : 'application/json', 'Accept' : 'application/json'}

print("Gen isimleri araniyor...")
try:
    r = requests.post(server+ext, headers=headers, json={'ids': genes})
    decoded = r.json()
    for g in genes:
        if g in decoded and decoded[g] is not None:
            name = decoded[g].get('display_name', 'Bilinmiyor')
            desc = decoded[g].get('description', '')
            print(f"{g}: {name}")
        else:
            print(f"{g}: Bulunamadi")
except Exception as e:
    print(f"Hata: {e}")
