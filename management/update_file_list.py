import os, json
from glob import glob
from bs4 import BeautifulSoup

def get_title(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        html_content = f.readlines()
    if html_content[0].startswith('---'):
        return html_content[2][6:].strip().strip('"')
    else:
        bs = BeautifulSoup(open(filename, 'r', encoding='utf-8').read(), 'html.parser')
        return bs.find('h1', {'class': 'page-title'}).text
    
def run(dirname):
    json_dic = {"htmls": []}
    html_paths = glob(os.path.join(dirname, '**/*.html'), recursive=True)
    # html_list = [os.path.join(dirname, f) for f in os.listdir(dirname)\
    #     if f.endswith('.html') and not f.endswith('-index.html')]
    for html_path in html_paths:
        if html_path.endswith('-index.html'):
            continue
        html_path = html_path.replace('\\', '/')
        category = html_path.split('/')[1]
        href = os.path.join('/', html_path)
        ctime = os.path.getctime(html_path)
        title = get_title(html_path)
        json_dic['htmls'].append({
            'title': title.strip('"'),
            'time': ctime,
            'href': href,
            'category': category
        })
    with open(f'management/{dirname}.json', 'w', encoding='utf-8') as f:
        json.dump(json_dic, f, indent=4, ensure_ascii=False)
   


if __name__ == '__main__':
    dirname_list = ['notes', 'papers']
    for dirname in dirname_list:
        run(dirname)