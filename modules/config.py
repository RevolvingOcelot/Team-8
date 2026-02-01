# S&P 500 stocks with long trading history
# Paper uses N=194 stocks from 1985-2016
SP500_TICKERS = [
    'CMCSA', 'DIS', 'F', 'GPC', 'GT', 'HAS', 'HD', 'HRB', 'IPG',
    'LEG', 'LEN', 'LOW', 'MAT', 'MCD', 'NKE', 'SHW', 'TGT',
    'VFC', 'WHR', 'ADM', 'CAG', 'CL', 'CPB', 'CVS', 'GIS', 'HRL',
    'HSY', 'K', 'KMB', 'KO', 'KR', 'MKC', 'MO', 'SYY', 'TAP', 'TSN',
    'WMT', 'APA', 'COP', 'CVX', 'HAL', 'HP', 'MUR',
    'NBR', 'SLB', 'VLO', 'WMB', 'XOM', 'AFL', 'AIG', 'AON',
    'AXP', 'BAC', 'BBT', 'BEN', 'BK', 'CB', 'CINF', 'CMA', 'C', 'EFX',
    'FHN', 'HBAN', 'HST', 'JPM', 'L', 'LNC', 'MMC',
    'MTB', 'PSA', 'SLM', 'TRV', 'USB', 'VNO', 'WFC', 'WY', 'ZION',
    'ABT', 'AET', 'AMGN', 'BAX', 'BDX', 'BMY', 'CAH', 'CI', 'HUM',
    'JNJ', 'LLY', 'MDT', 'MRK', 'SYK', 'THC', 'TMO', 'UNH',
    'AVY', 'BA', 'CAT', 'CMI', 'CSX', 'CTAS', 'DE', 'DHR', 'DOV',
    'EMR', 'ETN', 'EXPD', 'FDX', 'FLS', 'GD', 'GE', 'GLW', 'GWW', 'HON',
    'ITW', 'LMT', 'LUV', 'MAS', 'MMM', 'ROK', 'TXT',
    'UNP', 'AAPL', 'ADI', 'ADP', 'AMAT', 'AMD', 'HPQ',
    'IBM', 'INTC', 'KLAC', 'LRCX', 'MSI', 'MU', 'TXN', 'WDC', 'XRX',
    'AA', 'APD', 'BMS', 'CLF', 'DD', 'ECL', 'FMC', 'IFF', 'IP',
    'NEM', 'PPG', 'VMC', 'T', 'VZ', 'AEP', 'CMS', 'CNP',
    'D', 'DTE', 'ED', 'EIX', 'EQT', 'ETR', 'EXC', 'NEE', 'NI',
    'PNW', 'SO', 'WEC', 'XEL'
]

# Parameters from paper: M=40 days, Δ=20 days shift
START_DATE = '1985-01-02'  # Adjust based on data availability
END_DATE = '2025-12-31'
EPOCH_SIZE = 40   # M = 40 days
SHIFT = 20        # Δ = 20 days

MARKET_EVENTS = {
    'crashes': [
        ('1987-10-19', '1987-10-30', 'Black Monday'),
        ('1989-10-13', '1989-10-20', 'Friday 13th Mini Crash'),
        ('1990-01-01', '1990-12-31', 'Early 90s Recession'),
        ('1997-10-27', '1997-11-15', 'Asian Financial Crisis'),
        ('2000-03-10', '2000-04-14', 'Dot-com Crash'),
        ('2001-09-11', '2001-09-30', '9/11 Financial Crisis'),
        ('2002-09-01', '2002-10-15', 'Stock Market Downturn 2002'),
        ('2008-09-15', '2008-11-30', 'Lehman Brothers Crash'),
        ('2010-05-06', '2010-05-20', 'DJ Flash Crash'),
        ('2011-03-11', '2011-03-25', 'Tsunami/Fukushima'),
        ('2011-08-08', '2011-08-31', 'August 2011 Markets Fall'),
        ('2015-08-24', '2015-09-15', 'Chinese Black Monday'),
        ('2018-02-02', '2018-02-09', 'Volatility spike'),
        ('2020-02-20', '2020-03-23', 'COVID Crash'),
    ],
    'bubbles': [
        ('1999-01-01', '2000-03-09', 'Dot-com Bubble'),
        ('2005-01-01', '2007-10-01', 'US Housing Bubble'),
        ('2021-01-01', '2021-11-15', 'Post-COVID Rally'),
    ]
}

CRISIS_PERIODS = [
    ('1987-10-19', '1987-10-30', 'Black Monday'),
    ('1997-10-27', '1997-11-15', 'Asian Crisis'),
    ('2000-03-10', '2000-04-14', 'Dot-com Crash'),
    ('2001-09-11', '2001-09-30', '9/11'),
    ('2008-09-15', '2008-11-30', 'Lehman'),
    ('2011-08-08', '2011-08-31', 'Aug 2011'),
    ('2015-08-24', '2015-09-15', 'China'),
]

MAJOR_CRASHES = [
    ('1987-10-19', 'Black\nMonday'),
    ('2000-03-10', 'Dot-com'),
    ('2008-09-15', 'Lehman'),
]

PHASE_MARKERS = {
    'Crash': ('^', 'red', 50),
    'Type-1': ('D', 'deepskyblue', 40),
    'Type-2': ('s', 'blue', 40),
    'Anomaly': ('o', 'green', 40),
    'Normal': ('o', 'gray', 15),
}

PHASE_THRESHOLDS = {
    "H_HM_low_q": 0.15,
    "H_HM_high_q": 0.85,
    "H_HGR_low_q": 0.15,
    "H_HGR_high_q": 0.85,
    "crash_hgr_mid_q": 0.50,
    "anomaly_q": 0.30,
    "type2_hm_mid_q": 0.50,
}

PHASE_WOE_POINT = (0.5, 0.005)

PHASE_TRANSITIONS = {
    "crash": ("2008-09-15", "r--", "darkred"),
    "bubble": ("2000-03-10", "b--", "darkblue"),
}

PHASE_PLOT_LIMITS = {
    "xlim": (1e-4, 1e0),
    "ylim": (1e-4, 1e0),
    "legend_loc": "lower left",
}

COLORS_3D = {
    'Crash': 'red',
    'Type-1': 'deepskyblue',
    'Type-2': 'blue',
    'Anomaly': 'green',
    'Normal': 'gray',
}

CRASH_EVENTS = [
    ("1987-10-19", "Black Monday"),
    ("1997-10-27", "Asian Crisis"),
    ("2000-03-10", "Dot-com Crash"),
    ("2001-09-11", "9/11"),
    ("2008-09-15", "Lehman"),
    ("2010-05-06", "Flash Crash"),
    ("2011-08-08", "Aug 2011"),
    ("2015-08-24", "China Black Monday"),
    ("2020-02-20", "COVID"),
]