import numpy as np
import pandas as pd
import quandl
import re

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def old_build_dataset():
    data = pd.read_csv('data/dataset.csv')
    data.replace('null', np.nan, inplace=True)

    data.fillna(method='ffill', inplace=True)

    data['PREVCLSSP500'] = pd.to_numeric(data['PREVCLSSP500'])
    data['PREVCLSDJI'] = pd.to_numeric(data['PREVCLSDJI'])
    data['PREVCLSBRENT'] = pd.to_numeric(data['PREVCLSBRENT'])
    data['GOLD'] = pd.to_numeric(data['GOLD'])
    data['N225'] = pd.to_numeric(data['N225'])
    data['HSI'] = pd.to_numeric(data['HSI'])
    data['OSEBXOPN'] = pd.to_numeric(data['OSEBXOPN'])

    data = data.iloc[::-1]

    data['PREVCLSSP500'] = data['PREVCLSSP500'].pct_change()
    data['PREVCLSDJI'] = data['PREVCLSDJI'].pct_change()
    data['PREVCLSBRENT'] = data['PREVCLSBRENT'].pct_change()
    data['GOLD'] = data['GOLD'].pct_change()
    data['N225'] = data['N225'].pct_change()
    data['HSI'] = data['HSI'].pct_change()
    data['OSEBXOPN'] = data['OSEBXOPN'].pct_change()

    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.drop(data.index[0:2], inplace=True)

    data['OSEBXSGN'] = np.sign(data['OSEBXOPN'])
    data['OSEBXABS'] = np.abs(data['OSEBXOPN'])

    data.drop('DATE', 1, inplace=True)
    data.to_csv('data/train.csv', index=False)

    # data.tail(10)
    # data.head(10)


def load_data(filename, seq_len):
    # data = np.genfromtxt(filename, delimiter=',')
    data = np.genfromtxt('data/train.csv', delimiter=',', skip_header=1)

    sequence_length = seq_len
    X = []
    y = []
    for index in range(len(data) - sequence_length):
        X.append(data[index: index + sequence_length, :6])
        # y.append(data[index: index + sequence_length, 6])
        y.append(data[index + sequence_length-1, 6])

    X = np.array(X)
    y = np.array(y)

    row = int(round(0.8 * X.shape[0]))

    x_train = X[:row]
    y_train = y[:row]

    x_test = X[row:]
    y_test = y[row:]

    # x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 6))
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 6))

    return [x_train, y_train, x_test, y_test]


def gold():
    print('Fetching gold prices')
    gold = pd.read_csv('http://www.quandl.com/api/v1/datasets/LBMA/GOLD.csv')
    gold.head()
    gold['Date'] = pd.to_datetime(gold['Date'], format='%Y-%m-%d')
    gold.set_index('Date', inplace=True)
    gold.index.name = 'quote_date'
    gold.drop(['GBP (AM)', 'GBP (PM)', 'EURO (AM)', 'EURO (PM)'], axis=1, inplace=True)
    curr_cols = ['gold_{}'.format(re.sub(r'[^A-Za-z0-9]', '', col).lower()) for col in list(gold.columns)]
    gold.columns = curr_cols
    gold.head()
    return gold


def brent():
    print('Fetching brent oil prices')
    brent = quandl.get("COM/OIL_BRENT", authtoken="WM8sJvsKsnrHmRA_TkZD")
    brent_cols = ['brent_{}'.format(col) for col in list(brent.columns)]
    brent.columns = brent_cols
    brent.index.name = 'quote_date'
    brent.head()
    return brent


def usd_nok():
    print('Fetching USD/NOK prices')
    usd_nok = pd.read_csv('http://www.netfonds.no/quotes/paperhistory.php?paper=USDNOK.FXSX&csv_format=csv')
    usd_nok['quote_date'] = pd.to_datetime(usd_nok['quote_date'], format='%Y%m%d')
    usd_nok.set_index('quote_date', inplace=True)
    usd_nok.drop([c for c in usd_nok.columns if c not in ['close']], axis=1, inplace=True)
    curr_cols = ['usd_nok_{}'.format(col) for col in list(usd_nok.columns)]
    usd_nok.columns = curr_cols
    usd_nok.head()
    return usd_nok


def osebx():
    print('Fetchin OSEBX prices')
    osebx = pd.read_csv('http://www.netfonds.no/quotes/paperhistory.php?paper=OSEBX.OSE&csv_format=csv')
    osebx.head()
    osebx['quote_date'] = pd.to_datetime(osebx['quote_date'], format='%Y%m%d')
    osebx.set_index('quote_date', inplace=True)
    osebx.drop([c for c in list(osebx.columns) if c not in ['open']], axis=1, inplace=True)
    curr_cols = ['osebx_{}'.format(col) for col in list(osebx.columns)]
    osebx.columns = curr_cols
    osebx.tail()
    return osebx


def build_dataset():
    papers = [brent, gold, usd_nok, osebx]
    df = pd.concat([f() for f in papers], axis=1)
    # df = df[np.isfinite(df['osebx_open'])]
    df.fillna(method='ffill', inplace=True)
    # df.dropna(axis=0, how='any', inplace=True)
    df.dropna(how='any', inplace=True)
    df['osebx_open'] = df['osebx_open'].shift(-1)
    df.tail()

    df.to_csv('data/data.csv')


if __name__ == '__main__':
    build_dataset()
    x_train, y_train, x_test, y_test = load_data('data/train.csv', 10)


