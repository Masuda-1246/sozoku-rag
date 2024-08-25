import requests
from bs4 import BeautifulSoup

# ターゲットURL
base_url = 'https://www.nta.go.jp/law/tsutatsu/kihon/sisan/sozoku2/01.htm'

# URLからHTMLを取得
response = requests.get(base_url)
soup = BeautifulSoup(response.content, "html.parser")

# ページのタイトルを取得
page_title = soup.find('title').get_text()

# ページ内のすべてのリンクを取得し、完全なURLを生成
links = ["https://www.nta.go.jp" + anchor.get('href') for anchor in soup.find_all('a')]

# kihonを含むリンクのみをフィルタリングし、ハッシュ(#)以降を削除して一意のリンクをセットに保存
filtered_links = {link.split('#')[0] for link in links if 'kihon' in link}

# リンクをソート
sorted_links = sorted(filtered_links)

# リンクをファイルに書き込む
with open('outputs/sozoku.txt', 'w') as output_file:
    for link in sorted_links:
        output_file.write(link + '\n')
