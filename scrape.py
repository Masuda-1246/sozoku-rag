import requests
from bs4 import BeautifulSoup
import time

# ファイルからURLリストを読み込む
urls = []
with open('outputs/sozoku.txt', 'r') as file:
    urls = [line.strip() for line in file]

# 出力ファイルを開く
with open('outputs/sozoku.csv', 'w') as output_file:
    # CSVヘッダーを書き込む
    output_file.write('title,text,url\n')

    # 各URLに対して処理を行う
    for url in urls:
        # URLに対してリクエストを送信し、HTMLを取得する
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # ページのタイトルを取得する
        page_title = soup.find('title').get_text()

        # 全てのh2タグを取得し、対応するpタグのテキストを取得する
        h2_tags = soup.find_all('h2')
        for h2 in h2_tags:
            h2_text = h2.get_text().strip()
            p_tag = h2.find_next_sibling('p')
            
            # pタグが存在する場合に、そのテキストを取得・整形する
            if p_tag:
                p_text = p_tag.get_text().strip().replace('\u3000', '').replace('\n', '').replace(',', '，')

                # タイトル、テキスト、URLをCSV形式で出力ファイルに書き込む
                full_title = page_title + h2_text
                csv_line = f'{full_title},{p_text},{url}\n'
                output_file.write(csv_line)

                # デバッグ用にコンソールにも出力
                print(csv_line)

        # サーバーに負荷をかけないように1秒の間隔を空ける
        time.sleep(1)
