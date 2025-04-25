def json2csv(json_file, csv_file):
    import json
    import csv

    with open(json_file, 'r') as f:
        data = json.load(f)
    print(len(data['train']), len(data['val']), len(data['test']))
    print(data['train'][0])

    print(data.keys())

    csv_names = ['train', 'val', 'test']
    for csv_name in csv_names:
        csv_file = csv_file.split('.')[0] + f'_{csv_name}.csv'
        with open(csv_file, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['image', 'label', 'classes'])
            for row in data[csv_name]:
                writer.writerow([row[0], row[1], row[2]])


if __name__ == '__main__':
    root = '/home/zmz/datasets/Flowers102'
    json2csv(f'{root}/split_zhou_OxfordFlowers.json', f'{root}/labels.csv')
