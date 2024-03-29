import os, json

def get_title(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        html_content = f.readlines()
        return html_content[2].split(':')[-1].strip() 
    
def run(dirname):
    json_dic = {"htmls": []}
    html_list = [os.path.join(dirname, f) for f in os.listdir(dirname)\
        if f.endswith('.html') and not f.endswith('-index.html')]
    for html in html_list:
        href = os.path.join('/', html)
        ctime = os.path.getctime(html)
        title = get_title(html)
        json_dic['htmls'].append({
<<<<<<< HEAD
            'title': title.strip('"'),
            'time': ctime,
            'href': href.replace('\\', '/')
=======
            'title': title,
            'time': ctime,
            'href': href
>>>>>>> 2c65f3f7413451b12a2171db20494ce985bac67a
        })
    with open(f'management/{dirname}.json', 'w', encoding='utf-8') as f:
        json.dump(json_dic, f, indent=4, ensure_ascii=False)
   


if __name__ == '__main__':
    dirname_list = ['notes', 'papers']
    for dirname in dirname_list:
        run(dirname)