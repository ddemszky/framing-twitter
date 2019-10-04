import tweepy
from requests.exceptions import Timeout, ConnectionError
import ssl
import os
import sys


oauth = int(sys.argv[1])
# Consumer keys and access tokens, used for OAuth
consumer_keys = ['u7k40l5CAuntCyZ7SgGRyQ6EA', 'aXYeSMWaKbVNUfhZhFYTJjRDF', 'J2gXRArrezEqHzkifoSFEF3Ke', 'nHu1N2Bi1LN7FvI6ozcRfUs7a',
                 'jbjsSMDBt9hrYbiZfWhk4JItJ','ntayrvBTg18UYytRBwmur2apc','6PnVkAPLgKv6A0DVA6VXQTzFF', 'Iu0baHGRVQnB5TMZUZnTsfadS']
consumer_secrets = ['nAgRG0g5SlQ20kB4wSq5H0dzotEVPqpQ4Ubn3RFNLYzw6Rnbsq', 'coJ5n98VxlQ29Rm4McssidlB5hm3bhDtQIemCPW7e6sIR6QMsU',
                    '8gEsTWdmbuFzeZkMkj3rMkJkXAstJlL8arRKKTfZIWXjTL3eyM', 'MV8V9NBbIgdyoHnHouJDILMhk6bWp9Oyqdt9HPY8fTfijzMklj',
                    '3jf6FrnbwTdPOC5IT9XRpvVptadM0Jco22bMiZ5dIcTGzMcLi9','dd7go4ONfwbYp2jciZ1fyoqAZZtWALBbauGE1BgvDAhtOMO6Fq',
                    'B11YsHzAvUZjLzb5HvYMFwU4D28i15DRlgP4pnTiYECQeMulYE', 'fJ6l6qpBUqDqShM7JSVUfQyW7Qbw95J4p9x241NvoT46ZXDA0X']
access_tokens = ['1871986214-kKKrDycDnMSdoxF1cN28O8bCZsSoTwhpXYYBqxa', '1871986214-zHAZyu92RmWwF0mWjJhODvfalfUv8XTrDSlYKGQ',
                 '1871986214-3N7uoeEHtuGFb4zpcv57ebeCeBaLOFUSBMkA8La', '1871986214-c0SmlFBvhRcy8iJW1C24hktq61WTnohKEg1cRHL',
                 '933345003639844865-2vMindgotPeayg4yzIaklc3UxcdS83Y','933345003639844865-iLlSM3XODwQyAAPH77Mqj5SJRzSSnX4',
                 '933345003639844865-q35Abb5DDCArTuDg4it3sFlrtkBh9AV', '933345003639844865-VQtbjTZwfgNRLCyC7LZuBAKLsd9y7cR']
access_token_secrets = ['Flay9STENB19tMvmW9S3NL8EbzvJKxF6dGwyUuCGgL14D', 'MII6xnhEs58RArEGoaE2mNhI8YUC6HFUkToZT971CHCS3',
                        'XwuVABG1NKDyVHdDU7CWl1nnjgd7DOHwthM3DHLOfjca4', 'x6iYT61AMQueArK94WxI7CZgQLLOMvIdzPYbyB087IbEJ',
                        'EezWH4Q3beIhdj5tp1r4fDKvfXeEel3IdVkrg1BuKe9GC','61v4mrDhsjJtYGikYCZM9oyhdNHHAiJzze13mVONE9j2Y',
                        'wCoJ0zDgis31EqCo4aVztikjcBYkHIRPAmimZ0NSheRx5', 'SvIF0yxxz0Xz8a58ew9CsfFgQQoYASUtfk54NtJO2N7XF']


# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_keys[oauth], consumer_secrets[oauth])
auth.set_access_token(access_tokens[oauth], access_token_secrets[oauth])

# Creation of the actual interface, using authentication
api = tweepy.API(auth, retry_count=3, retry_delay=5, retry_errors=set([104, 401, 404, 500, 503]), timeout=2000, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

users = []

with open('users.txt', 'r') as f:
    users = f.read().splitlines()

total = len(users)


i = 0
with open('active_ids.txt', 'w') as f:
    while i < (total / 100) + 1:
        try:
            f.write('\n'.join([str(u.id) for u in api.lookup_users(user_ids=users[i*100:min((i+1)*100, total)])]))
            f.write('\n')
            print 'getting users batch:', i
            i += 1
        except tweepy.TweepError as e:
            print e
            pass

