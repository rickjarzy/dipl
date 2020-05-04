import os
import requests
import urllib
import urllib3
import certifi


from bs4 import BeautifulSoup
from base64 import b64encode
from datetime import date
from urllib import request
from http.cookiejar import CookieJar


class SessionWithHeaderRedirection(requests.Session):
    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):
        super(SessionWithHeaderRedirection, self).__init__()

        self.auth = (username, password)

    # Overrides from the library to keep headers when redirected to or from
    # the NASA auth host.

    def rebuild_auth(self, prepared_request, response):
        headers = prepared_request.headers

        url = prepared_request.url

        if 'Authorization' in headers:
            original_parsed = requests.utils.urlparse(response.request.url)
            redirect_parsed = requests.utils.urlparse(url)

        if (original_parsed.hostname != redirect_parsed.hostname) and (redirect_parsed.hostname != self.AUTH_HOST) and (original_parsed.hostname != self.AUTH_HOST):
            del headers["Authorization"]




if __name__ == "__main__":

    kacheln = ["h18v04", "h18v03", "h19v03", "h19v04"]
    mcd43a2 = "MCD43A2"
    mcd43a4 = "MCD43A4"

    topics = [mcd43a2, mcd43a4]
    username = "pauljarzy"
    password = "summerINT2013#!"

    session = SessionWithHeaderRedirection(username, password)

    doys = [i for i in range(1,362,8)]
    years = [i for i in range(2000,2021,1)]

    print("start download")
    version = ".006"
    root_server = r"https://e4ftl01.cr.usgs.gov/MOTA/"
    goal_drive = r"R:\modis\v6\hdf"

    userpwd = "{us}:{pw}".format(us=username, pw=password)
    userAndPass = b64encode(str.encode(userpwd)).decode("ascii")
    print("USER PASSWORD: ", userAndPass)
    http_header = {'Authorization': 'Basic %s' % userAndPass}



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

            for kachel in kacheln:

                for topic in topics:

                    # create urls to download
                    url = root_server + topic + version + "/{}.{}.{}/".format(calc_date.year, month, day)
                    print("# Processing : %d of %d" % (doy, year), "  \n  url:",url)
                    ret_html = requests.get(url)

                    data = ret_html.content

                    soup = BeautifulSoup(data, "html.parser")
                    a_list = soup.find_all('a')
                    matches = [i for i in a_list if kachel in str(i)]

                    for match in matches:

                        if ".jpg" in match.text:
                            print("  skipping jpg")
                            continue
                        print("  Processing tile: ", kachel)
                        print("  processing file: ", match.text)

                        response = session.get(url+match.text, stream=True)
                        print("  response status code: ",response.status_code)

                        outdir = os.path.join(goal_drive, topic, kachel, match.text)
                        print("  outdir: ", outdir)


                        response.raise_for_status()


                        if os.path.exists(outdir) and os.stat(outdir).st_size > 9000:
                            print("  file exists")
                            continue
                        else:
                            print("  start downloading ...")
                            cou = 0
                            with open(outdir, "wb") as download_file:
                                for chunk in response.iter_content(chunk_size=4096*4096):
                                    print("download chunk ", cou)
                                    download_file.write(chunk)

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

