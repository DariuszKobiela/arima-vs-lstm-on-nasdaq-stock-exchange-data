with open('TMP_companies.txt') as f:
    lines = f.read()
    companies_list = lines.split('\n')
    print(companies_list)
    f.close()