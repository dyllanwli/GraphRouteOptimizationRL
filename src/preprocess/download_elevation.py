import requests  # get the requsts library from https://github.com/requests/requests

from tqdm import tqdm
# overriding requests.Session.rebuild_auth to mantain headers when redirected

class SessionWithHeaderRedirection(requests.Session):

    AUTH_HOST = 'urs.earthdata.nasa.gov'

    def __init__(self, username, password):

        super().__init__()

        self.auth = (username, password)

   # Overrides from the library to keep headers when redirected to or from

   # the NASA auth host.

    def rebuild_auth(self, prepared_request, response):

        headers = prepared_request.headers

        url = prepared_request.url

        if 'Authorization' in headers:

            original_parsed = requests.utils.urlparse(response.request.url)

            redirect_parsed = requests.utils.urlparse(url)

            if (original_parsed.hostname != redirect_parsed.hostname) and redirect_parsed.hostname != self.AUTH_HOST and original_parsed.hostname != self.AUTH_HOST:

                del headers['Authorization']

        return


# create session with the user credentials that will be used to authenticate access to the data




session = SessionWithHeaderRedirection(username, password)




if __name__ == "__main__":
    # the url of the file we wish to retrieve

    with open("/h/zk-persistent-volumnes/opentopodata/data/srtm30m_urls.txt") as f:
        urls = f.readlines()

    for url in tqdm(urls):
        url = url.strip()
        # extract the filename from the url to be used when saving the fil
        filename = url[url.rfind('/')+1:]
        try:
            # submit the request using the session

            response = session.get(url, stream=True)

            print(response.status_code)

            # raise an exception in case of http errors

            response.raise_for_status()

            # save the file

            with open(filename, 'wb') as fd:

                for chunk in response.iter_content(chunk_size=1024*1024):

                    fd.write(chunk)
        except requests.exceptions.HTTPError as e:

            # handle any errors here

            print(e)
        