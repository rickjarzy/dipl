
from bs4 import BeautifulSoup
import os
import requests
from base64 import b64encode
from datetime import date
import shutil
import urllib
import urllib3
import certifi
from downModis import downModis



if __name__ == "__main__":

    kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
    mcd43a2 = "MCD43A2"
    mcd43a4 = "MCD43A4"
    username = "pauljarzy"
    password = "summerINT2013#!"
    doys = [i for i in range(1,362,8)]
    years = [i for i in range(2000,2021,1)]

    print("start download")
    version = ".006"
    root_server = r"https://e4ftl01.cr.usgs.gov/MOTA/"

    goal_drive = r"M:\modis\v6\hdf"

    userpwd = "{us}:{pw}".format(us=username,
                                      pw=password)
    userAndPass = b64encode(str.encode(userpwd)).decode("ascii")
    print("USER PASSWORD: ", userAndPass)
    http_header = {'Authorization': 'Basic %s' % userAndPass}

    print("ca : ", certifi.where())
    urllib3_poolmanager = urllib3.PoolManager(ca_certs=certifi.where())

    for year in years:

        if year == 2000:
            doys = doys[7:]
        elif year == 2020:
            doys = doys[:14]
        else:
            doys = [i for i in range(1,362,8)]
        for doy in doys:
            calc_date = date.fromordinal(date(year,1,1).toordinal() + doy-1)

            print("date: {}.{}.{}".format(calc_date.year, calc_date.month, calc_date.day)," - doy: ", doy)

            if calc_date.month < 10:
                month = "0{}".format(calc_date.month)
            else:
                month = "{}".format(calc_date.month)

            if calc_date.day < 10:
                day = "0{}".format(calc_date.day)
            else:
                day = "{}".format(calc_date.day)

            url_a2 = root_server + mcd43a2 + version + "/{}.{}.{}/".format(calc_date.year, month, day)
            url_a4 = root_server + mcd43a4 + version + "/{}.{}.{}/".format(calc_date.year, month, day)
            urls = [url_a2, url_a4]

            print(url_a2)
            print(url_a4)

            for kachel in ["h18v03"]:

                for url in urls:

                    ret_html = requests.get(url)

                    data = ret_html.content

                    soup = BeautifulSoup(data, "html.parser")
                    a_list = soup.find_all('a')
                    matches = [i for i in a_list if kachel in str(i)]
                    cou = 0
                    for match in matches:
                        if cou == 2:
                            cou = 0
                            break
                        print(match.text)

                        with urllib3_poolmanager.request("GET", url, headers=http_header, preload_content=False) as response, open(goal_drive + "\\" + url.split("/")[4].split(".")[0] + "\\" + kachel + "\\" + str(match.text), "wb") as out_file:
                            print("response status: ", response.status)
                            shutil.copyfileobj(response, out_file)
                            response.release_conn()


                        # response = requests.get(url + match.text)
                        # print("status code: ", response.status_code)
                        #
                        # with open(goal_drive + "\\" + mcd43a2 + "\\" + kachel + "\\" + match.text, "wb") as download:
                        #     shutil.copyfileobj(response.content, download)

                        # fileSave =  open(goal_drive + "\\" + url.split("/")[4].split(".")[0] + "\\" + kachel + "\\" + str(match.text), "wb")
                        #
                        # orig_size = None
                        # try:  # download and write the file
                        #     req = urllib.request.Request(url, headers=http_header)
                        #     http = urllib.request.urlopen(req)
                        #     orig_size = http.headers['Content-Length']
                        #     fileSave.write(http.read())
                        #     fileSave.close()
                        # except:
                        #     print("geht a net")

                        cou += 1




    html_page_data = "https://e4ftl01.cr.usgs.gov/MOTA/MCD43A2.006/2000.02.24/"
    ret_html = requests.get(html_page_data)
    data = ret_html.content

    # beautify data from requests
    soup = BeautifulSoup(data, "html.parser")
    a_list = soup.find_all('a')

    print("len of a: ", len(a_list))
    print(a_list[3].text)
    print(str(type(a_list[0])))
    print(str(a_list[0]))
    matches = [i for i in a_list if kacheln[0] in str(i)]

    for match in matches:
        print(match.text)


    print("Programm ENDE")

