import re
import hashlib
from typing import Dict, List
import pandas as pd

# ========== Templates / keywords ==========

QUERY_TEMPLATE = {
    "positive": "{} stock price increased significantly",
    "negative": "{} stock price fell sharply",
}

FIN_KEYWORDS: List[str] = [
    # Earnings / financial reports
    "earnings", "revenue", "profit", "eps", "guidance", "forecast",
    "estimate", "beat", "miss", "margin", "sales", "cost", "expense",
    "fy", "quarter",
    # Business operations
    "production", "delivery", "demand", "supply", "shipment", "factory",
    "growth", "order", "contract", "deal", "partnership", "launch",
    "recall", "shutdown", "layoff", "hiring",
    # Executives / management
    "ceo", "cfo", "executive", "board", "leadership", "director",
    "resign", "appoint",
    # Legal & regulatory
    "lawsuit", "court", "probe", "investigation", "sec", "ftc", "fine",
    "ban", "antitrust", "regulation", "compliance", "penalty", "tariff",
    "sanction",
    # Capital markets / corporate actions
    "ipo", "buyback", "dividend", "merger", "acquisition", "m&a",
    "spinoff", "upgrade", "downgrade", "price target", "rating",
    "financing", "loan", "bond",
    # Macro & industry
    "inflation", "fed", "interest rate", "policy", "government",
    "economy", "labor", "strike", "chip", "supply chain", "commodity",
    "oil", "gold",
]


# ========== Publisher weights / alias ==========

PUBLISHER_WEIGHTS: Dict[str, float] = {
    # Tier 1: Top Tier Finance
    "bloomberg": 1.00,
    "wallstreetjournal": 1.00,
    "reuters": 1.00,
    "financialtimes": 1.00,
    "cnbc": 1.00,
    "cnn": 1.00,
    "forbes": 1.00,
    "economist": 1.00,
    "institutionalinvestor": 1.00,
    "barrons": 1.00,
    "nikkei": 1.00,
    "scmp": 1.00,
    "globeandmail": 1.00,
    "australianfinancialreview": 1.00,
    "bloombergbusinessweek": 1.00,
    # Tier 2
    "yahoofinance": 0.75,
    "marketwatch": 0.75,
    "businessinsider": 0.75,
    "investopedia": 0.75,
    "thestreet": 0.75,
    "msnmoney": 0.75,
    "morningstar": 0.75,
    "economictimes": 0.75,
    "moneycontrol": 0.75,
    "citywire": 0.75,
    "financialnews": 0.75,
    # Tier 3
    "seekingalpha": 0.55,
    "motleyfool": 0.55,
    "valuewalk": 0.55,
    "realvision": 0.55,
    "finextra": 0.55,
    "fintechmagazine": 0.55,
}

PUBLISHER_ALIAS: Dict[str, str] = {
    # Tier 1
    "bloomberg": "bloomberg",
    "bloombergtv": "bloomberg",
    "bloombergbusiness": "bloomberg",
    "bloombergbusinessweek": "bloombergbusinessweek",
    "bloombergnews": "bloomberg",
    "wallstreetjournal": "wallstreetjournal",
    "wsj": "wallstreetjournal",
    "wallstjournal": "wallstreetjournal",
    "wallstreetjnl": "wallstreetjournal",
    "reuters": "reuters",
    "reutersnews": "reuters",
    "financialtimes": "financialtimes",
    "ft": "financialtimes",
    "fttimes": "financialtimes",
    "cnbc": "cnbc",
    "cnbctv": "cnbc",
    "cnn": "cnn",
    "cnnnews": "cnn",
    "forbes": "forbes",
    "economist": "economist",
    "theeconomist": "economist",
    "institutionalinvestor": "institutionalinvestor",
    "barrons": "barrons",
    "barron": "barrons",
    "nikkei": "nikkei",
    "nikkeiasia": "nikkei",
    "scmp": "scmp",
    "southchinamorningpost": "scmp",
    "globeandmail": "globeandmail",
    "theglobeandmail": "globeandmail",
    "australianfinancialreview": "australianfinancialreview",
    "financialreview": "australianfinancialreview",
    "bloombergbusinessweek": "bloombergbusinessweek",
    # Tier 2
    "yahoofinance": "yahoofinance",
    "yahoo": "yahoofinance",
    "yahoonews": "yahoofinance",
    "marketwatch": "marketwatch",
    "market watch": "marketwatch",
    "businessinsider": "businessinsider",
    "insider": "businessinsider",
    "investopedia": "investopedia",
    "thestreet": "thestreet",
    "the street": "thestreet",
    "msnmoney": "msnmoney",
    "msn": "msnmoney",
    "morningstar": "morningstar",
    "economictimes": "economictimes",
    "theeconomictimes": "economictimes",
    "moneycontrol": "moneycontrol",
    "citywire": "citywire",
    "financialnews": "financialnews",
    "fnlondon": "financialnews",
    # Tier 3
    "seekingalpha": "seekingalpha",
    "seeking alpha": "seekingalpha",
    "motleyfool": "motleyfool",
    "themotleyfool": "motleyfool",
    "valuewalk": "valuewalk",
    "realvision": "realvision",
    "finextra": "finextra",
    "fintechmagazine": "fintechmagazine",
    "fintech": "fintechmagazine",
}

DEFAULT_UNKNOWN_SCORE = 0.20


# ========== Company profiles ==========

COMPANY_PROFILES = {
    "TSLA": {
        "canonical": "Tesla, Inc.",
        "tickers": ["NASDAQ:TSLA"],
        "org_names": [
            "Tesla",
            "Tesla, Inc.",
            "Tesla Motors",
            "Tesla Motors, Inc.",
        ],
        "people": {
            "founders": ["Martin Eberhard", "Marc Tarpenning"],
            "executives": [
                "Elon Musk",
                "Robyn Denholm",
            ],
        },
        "brands_units": [
            "Tesla Energy",
            "Gigafactory",
            "NACS",
        ],
        "products": [
            "Model S",
            "Model X",
            "Model 3",
            "Model Y",
            "Cybertruck",
            "Tesla Semi",
            "Powerwall",
            "Megapack",
            "Solar Roof",
            "Tesla Solar",
        ],
        "industry": ["Automotive", "Renewable energy"],
        "locations": {
            "hq": ["Austin", "Texas", "United States"],
            "regions": [
                "North America",
                "Europe",
                "East Asia",
                "Middle East",
                "Oceania",
                "Southeast Asia",
                "Indian Subcontinent",
            ],
        },
        "urls": {
            "site": "https://www.tesla.com/",
            "ir": "https://ir.tesla.com/",
            "wiki": "https://en.wikipedia.org/wiki/Tesla,_Inc.",
        },
    },
    "NVDA": {
        "canonical": "NVIDIA Corporation",
        "tickers": ["NASDAQ:NVDA"],
        "org_names": [
            "NVIDIA",
            "NVIDIA Corp.",
            "NVIDIA Corporation",
            "nVidia",
        ],
        "people": {
            "founders": ["Jensen Huang", "Chris Malachowsky", "Curtis Priem"],
            "executives": ["Jensen Huang"],
        },
        "brands_units": [
            "GeForce",
            "Quadro",
            "Tesla (GPU)",
            "DGX",
            "NVIDIA AI Enterprise",
            "CUDA",
            "Grace",
            "Blackwell",
            "Omniverse",
        ],
        "products": [
            "H100",
            "H200",
            "A100",
            "A800",
            "B100",
            "GB200",
            "DGX GH200",
            "GeForce RTX",
            "Jetson",
            "Triton Inference Server",
        ],
        "industry": ["Semiconductors", "AI Computing"],
        "locations": {
            "hq": ["Santa Clara", "California", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.nvidia.com/",
            "ir": "https://investor.nvidia.com/",
            "wiki": "https://en.wikipedia.org/wiki/Nvidia",
        },
    },
    "AAPL": {
        "canonical": "Apple Inc.",
        "tickers": ["NASDAQ:AAPL"],
        "org_names": [
            "Apple",
            "Apple Inc.",
        ],
        "people": {
            "founders": ["Steve Jobs", "Steve Wozniak", "Ronald Wayne"],
            "executives": ["Tim Cook"],
        },
        "brands_units": [
            "App Store",
            "iCloud",
            "Apple Music",
            "Apple TV+",
            "Beats",
            "Apple Pay",
        ],
        "products": [
            "iPhone",
            "iPad",
            "Mac",
            "Apple Watch",
            "AirPods",
            "Vision Pro",
            "M-series chips",
        ],
        "industry": ["Consumer Electronics", "Software", "Services"],
        "locations": {
            "hq": ["Cupertino", "California", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.apple.com/",
            "ir": "https://investor.apple.com/",
            "wiki": "https://en.wikipedia.org/wiki/Apple_Inc.",
        },
    },
    "MSFT": {
        "canonical": "Microsoft Corporation",
        "tickers": ["NASDAQ:MSFT"],
        "org_names": [
            "Microsoft",
            "Microsoft Corporation",
            "MSFT",
        ],
        "people": {
            "founders": ["Bill Gates", "Paul Allen"],
            "executives": ["Satya Nadella"],
        },
        "brands_units": [
            "Windows",
            "Microsoft 365",
            "Azure",
            "Xbox",
            "Surface",
            "LinkedIn",
            "GitHub",
        ],
        "products": [
            "Windows",
            "Microsoft 365",
            "Azure",
            "Xbox",
            "Surface",
            "LinkedIn",
            "GitHub",
        ],
        "industry": [
            "Software",
            "Cloud computing",
            "Consumer electronics",
            "Gaming",
        ],
        "locations": {
            "hq": ["Redmond", "Washington", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.microsoft.com/",
            "ir": "https://www.microsoft.com/investor",
            "wiki": "https://en.wikipedia.org/wiki/Microsoft",
        },
    },
    "GOOGL": {
        "canonical": "Alphabet Inc.",
        "tickers": ["NASDAQ:GOOGL", "NASDAQ:GOOG"],
        "org_names": [
            "Alphabet",
            "Alphabet Inc.",
            "Google",
        ],
        "people": {
            "founders": ["Larry Page", "Sergey Brin"],
            "executives": ["Sundar Pichai"],
        },
        "brands_units": [
            "Google",
            "YouTube",
            "Android",
            "Google Cloud",
            "Waymo",
            "DeepMind",
            "Google Maps",
            "Gmail",
        ],
        "products": [
            "Google Search",
            "YouTube",
            "Android",
            "Google Chrome",
            "Google Cloud Platform",
            "Gmail",
            "Google Maps",
        ],
        "industry": [
            "Internet services",
            "Online advertising",
            "Cloud computing",
            "Artificial intelligence",
        ],
        "locations": {
            "hq": ["Mountain View", "California", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://abc.xyz/",
            "ir": "https://abc.xyz/investor/",
            "wiki": "https://en.wikipedia.org/wiki/Alphabet_Inc.",
        },
    },
    "AMZN": {
        "canonical": "Amazon.com, Inc.",
        "tickers": ["NASDAQ:AMZN"],
        "org_names": [
            "Amazon",
            "Amazon.com, Inc.",
            "Amazon.com",
        ],
        "people": {
            "founders": ["Jeff Bezos"],
            "executives": ["Andy Jassy", "Jeff Bezos"],
        },
        "brands_units": [
            "Amazon.com",
            "Amazon Web Services",
            "Prime Video",
            "Amazon Prime",
            "Kindle",
            "Alexa",
            "Whole Foods Market",
            "Twitch",
        ],
        "products": [
            "Amazon online marketplace",
            "Amazon Web Services (AWS)",
            "Prime membership",
            "Prime Video",
            "Kindle e-readers",
            "Fire TV",
            "Echo (Alexa)",
            "Twitch",
        ],
        "industry": [
            "E-commerce",
            "Cloud computing",
            "Digital streaming",
            "Consumer electronics",
        ],
        "locations": {
            "hq": ["Seattle", "Washington", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.amazon.com/",
            "ir": "https://ir.aboutamazon.com/",
            "wiki": "https://en.wikipedia.org/wiki/Amazon_(company)",
        },
    },
    "META": {
        "canonical": "Meta Platforms, Inc.",
        "tickers": ["NASDAQ:META"],
        "org_names": [
            "Meta",
            "Meta Platforms, Inc.",
            "Facebook",
        ],
        "people": {
            "founders": [
                "Mark Zuckerberg",
                "Eduardo Saverin",
                "Andrew McCollum",
                "Dustin Moskovitz",
                "Chris Hughes",
            ],
            "executives": ["Mark Zuckerberg"],
        },
        "brands_units": [
            "Facebook",
            "Instagram",
            "WhatsApp",
            "Messenger",
            "Meta Quest",
            "Reality Labs",
        ],
        "products": [
            "Facebook",
            "Instagram",
            "WhatsApp",
            "Messenger",
            "Meta Quest",
        ],
        "industry": [
            "Social media",
            "Online advertising",
            "Virtual reality",
        ],
        "locations": {
            "hq": ["Menlo Park", "California", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://about.meta.com/",
            "ir": "https://investor.fb.com/",
            "wiki": "https://en.wikipedia.org/wiki/Meta_Platforms",
        },
    },
    "BRK-B": {
        "canonical": "Berkshire Hathaway Inc.",
        "tickers": ["NYSE:BRK.B"],
        "org_names": [
            "Berkshire Hathaway",
            "Berkshire Hathaway Inc.",
        ],
        "people": {
            "founders": [],
            "executives": ["Warren Buffett"],
        },
        "brands_units": [
            "GEICO",
            "BNSF Railway",
            "Berkshire Hathaway Energy",
            "See's Candies",
            "Dairy Queen",
            "Duracell",
        ],
        "products": [
            "Insurance and reinsurance",
            "Railroad transportation",
            "Energy and utilities",
            "Manufacturing",
            "Retail",
        ],
        "industry": [
            "Conglomerate",
            "Insurance",
            "Investments",
        ],
        "locations": {
            "hq": ["Omaha", "Nebraska", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.berkshirehathaway.com/",
            "ir": "https://www.berkshirehathaway.com/reports.html",
            "wiki": "https://en.wikipedia.org/wiki/Berkshire_Hathaway",
        },
    },
    "LLY": {
        "canonical": "Eli Lilly and Company",
        "tickers": ["NYSE:LLY"],
        "org_names": [
            "Eli Lilly",
            "Eli Lilly and Company",
        ],
        "people": {
            "founders": ["Eli Lilly"],
            "executives": [],
        },
        "brands_units": [
            "Lilly Diabetes",
            "Lilly Oncology",
        ],
        "products": [
            "Pharmaceutical drugs",
        ],
        "industry": [
            "Pharmaceuticals",
            "Biotechnology",
        ],
        "locations": {
            "hq": ["Indianapolis", "Indiana", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.lilly.com/",
            "ir": "https://investor.lilly.com/",
            "wiki": "https://en.wikipedia.org/wiki/Eli_Lilly_and_Company",
        },
    },
    "ORCL": {
        "canonical": "Oracle Corporation",
        "tickers": ["NYSE:ORCL"],
        "org_names": [
            "Oracle",
            "Oracle Corp.",
            "Oracle Corporation",
        ],
        "people": {
            "founders": ["Larry Ellison", "Bob Miner", "Ed Oates"],
            "executives": ["Safra Catz", "Larry Ellison"],
        },
        "brands_units": [
            "Oracle Database",
            "Oracle Cloud Infrastructure",
            "Oracle Fusion Applications",
            "Java",
            "MySQL",
            "NetSuite",
        ],
        "products": [
            "Oracle Database",
            "Oracle Cloud Infrastructure (OCI)",
            "Oracle Fusion Applications",
            "Java platform",
            "MySQL",
            "NetSuite",
        ],
        "industry": [
            "Enterprise software",
            "Cloud computing",
            "Computer hardware",
        ],
        "locations": {
            "hq": ["Austin", "Texas", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.oracle.com/",
            "ir": "https://investor.oracle.com/",
            "wiki": "https://en.wikipedia.org/wiki/Oracle_Corporation",
        },
    },
    "V": {
        "canonical": "Visa Inc.",
        "tickers": ["NYSE:V"],
        "org_names": [
            "Visa",
            "Visa Inc.",
        ],
        "people": {
            "founders": [],
            "executives": [],
        },
        "brands_units": [
            "Visa",
            "Visa Electron",
            "Visa Debit",
            "Visa Plus",
        ],
        "products": [
            "Credit cards",
            "Debit cards",
            "Payment processing network",
        ],
        "industry": [
            "Financial services",
            "Payment processing",
        ],
        "locations": {
            "hq": ["San Francisco", "California", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.visa.com/",
            "ir": "https://investor.visa.com/",
            "wiki": "https://en.wikipedia.org/wiki/Visa_Inc.",
        },
    },
    "MA": {
        "canonical": "Mastercard Incorporated",
        "tickers": ["NYSE:MA"],
        "org_names": [
            "Mastercard",
            "Mastercard Incorporated",
        ],
        "people": {
            "founders": [],
            "executives": [],
        },
        "brands_units": [
            "Mastercard",
            "Maestro",
            "Cirrus",
        ],
        "products": [
            "Credit cards",
            "Debit cards",
            "Payment processing network",
        ],
        "industry": [
            "Financial services",
            "Payment processing",
        ],
        "locations": {
            "hq": ["Purchase", "New York", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.mastercard.com/",
            "ir": "https://investor.mastercard.com/",
            "wiki": "https://en.wikipedia.org/wiki/Mastercard",
        },
    },
    "JPM": {
        "canonical": "JPMorgan Chase & Co.",
        "tickers": ["NYSE:JPM"],
        "org_names": [
            "JPMorgan",
            "JPMorgan Chase",
            "JPMorgan Chase & Co.",
        ],
        "people": {
            "founders": [],
            "executives": ["Jamie Dimon"],
        },
        "brands_units": [
            "J.P. Morgan",
            "Chase",
            "JPMorgan Asset Management",
            "JPMorgan Securities",
        ],
        "products": [
            "Retail banking",
            "Investment banking",
            "Asset management",
            "Commercial banking",
            "Credit cards",
        ],
        "industry": [
            "Banking",
            "Financial services",
        ],
        "locations": {
            "hq": ["New York City", "New York", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.jpmorganchase.com/",
            "ir": "https://www.jpmorganchase.com/ir",
            "wiki": "https://en.wikipedia.org/wiki/JPMorgan_Chase",
        },
    },
    "WMT": {
        "canonical": "Walmart Inc.",
        "tickers": ["NYSE:WMT"],
        "org_names": [
            "Walmart",
            "Walmart Inc.",
        ],
        "people": {
            "founders": ["Sam Walton"],
            "executives": [],
        },
        "brands_units": [
            "Walmart",
            "Sam's Club",
            "Walmart Supercenter",
        ],
        "products": [
            "Discount retail",
            "Supercenters",
            "Warehouse clubs",
            "E-commerce",
        ],
        "industry": [
            "Retail",
            "E-commerce",
        ],
        "locations": {
            "hq": ["Bentonville", "Arkansas", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.walmart.com/",
            "ir": "https://stock.walmart.com/",
            "wiki": "https://en.wikipedia.org/wiki/Walmart",
        },
    },
    "XOM": {
        "canonical": "Exxon Mobil Corporation",
        "tickers": ["NYSE:XOM"],
        "org_names": [
            "ExxonMobil",
            "Exxon Mobil Corporation",
            "Exxon",
            "Mobil",
        ],
        "people": {
            "founders": [],
            "executives": [],
        },
        "brands_units": [
            "Exxon",
            "Mobil",
            "Esso",
        ],
        "products": [
            "Crude oil",
            "Natural gas",
            "Petrochemicals",
            "Refined petroleum products",
        ],
        "industry": [
            "Oil and gas",
            "Energy",
        ],
        "locations": {
            "hq": ["Spring", "Texas", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://corporate.exxonmobil.com/",
            "ir": "https://corporate.exxonmobil.com/investors",
            "wiki": "https://en.wikipedia.org/wiki/ExxonMobil",
        },
    },
    "HD": {
        "canonical": "The Home Depot, Inc.",
        "tickers": ["NYSE:HD"],
        "org_names": [
            "Home Depot",
            "The Home Depot, Inc.",
        ],
        "people": {
            "founders": ["Bernard Marcus", "Arthur Blank", "Ron Brill", "Pat Farrah"],
            "executives": [],
        },
        "brands_units": [
            "The Home Depot",
        ],
        "products": [
            "Home improvement retail",
            "Building materials",
            "Garden supplies",
            "Home appliances",
        ],
        "industry": [
            "Retail",
            "Home improvement",
        ],
        "locations": {
            "hq": ["Atlanta", "Georgia", "United States"],
            "regions": ["North America"],
        },
        "urls": {
            "site": "https://www.homedepot.com/",
            "ir": "https://ir.homedepot.com/",
            "wiki": "https://en.wikipedia.org/wiki/The_Home_Depot",
        },
    },
    "KO": {
        "canonical": "The Coca-Cola Company",
        "tickers": ["NYSE:KO"],
        "org_names": [
            "Coca-Cola",
            "The Coca-Cola Company",
        ],
        "people": {
            "founders": ["Asa Griggs Candler"],
            "executives": [],
        },
        "brands_units": [
            "Coca-Cola",
            "Sprite",
            "Fanta",
            "Dasani",
            "Minute Maid",
            "Powerade",
        ],
        "products": [
            "Soft drinks",
            "Bottled water",
            "Juices",
            "Sports drinks",
            "Tea and coffee beverages",
        ],
        "industry": [
            "Beverages",
            "Non-alcoholic drinks",
        ],
        "locations": {
            "hq": ["Atlanta", "Georgia", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.coca-colacompany.com/",
            "ir": "https://investors.coca-colacompany.com/",
            "wiki": "https://en.wikipedia.org/wiki/The_Coca-Cola_Company",
        },
    },
    "CVX": {
        "canonical": "Chevron Corporation",
        "tickers": ["NYSE:CVX"],
        "org_names": [
            "Chevron",
            "Chevron Corporation",
        ],
        "people": {
            "founders": [],
            "executives": [],
        },
        "brands_units": [
            "Chevron",
            "Texaco",
            "Caltex",
        ],
        "products": [
            "Crude oil",
            "Natural gas",
            "Refined petroleum products",
            "Petrochemicals",
        ],
        "industry": [
            "Oil and gas",
            "Energy",
        ],
        "locations": {
            "hq": ["San Ramon", "California", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.chevron.com/",
            "ir": "https://investor.chevron.com/",
            "wiki": "https://en.wikipedia.org/wiki/Chevron_Corporation",
        },
    },
    "MCD": {
        "canonical": "McDonald's Corporation",
        "tickers": ["NYSE:MCD"],
        "org_names": [
            "McDonald's",
            "McDonald's Corporation",
        ],
        "people": {
            "founders": ["Richard McDonald", "Maurice McDonald"],
            "executives": [],
        },
        "brands_units": [
            "McDonald's",
        ],
        "products": [
            "Fast food restaurants",
            "Burgers",
            "Fries",
            "Beverages",
        ],
        "industry": [
            "Fast food",
            "Restaurant",
        ],
        "locations": {
            "hq": ["Chicago", "Illinois", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.mcdonalds.com/",
            "ir": "https://corporate.mcdonalds.com/corpmcd/investors.html",
            "wiki": "https://en.wikipedia.org/wiki/McDonald%27s",
        },
    },
    "INTC": {
        "canonical": "Intel Corporation",
        "tickers": ["NASDAQ:INTC"],
        "org_names": [
            "Intel",
            "Intel Corporation",
        ],
        "people": {
            "founders": ["Gordon Moore", "Robert Noyce"],
            "executives": [],
        },
        "brands_units": [
            "Intel Core",
            "Intel Xeon",
            "Intel Arc",
            "Intel Foundry Services",
        ],
        "products": [
            "Microprocessors",
            "Chipsets",
            "Discrete GPUs",
            "Data center solutions",
        ],
        "industry": [
            "Semiconductors",
            "Computer hardware",
        ],
        "locations": {
            "hq": ["Santa Clara", "California", "United States"],
            "regions": ["Global"],
        },
        "urls": {
            "site": "https://www.intel.com/",
            "ir": "https://www.intc.com/",
            "wiki": "https://en.wikipedia.org/wiki/Intel",
        },
    },
}


TICKER_TO_NAME_MAP = {
    ticker: profile["canonical"] for ticker, profile in COMPANY_PROFILES.items()
}
companies_list = list(COMPANY_PROFILES.keys())


# ========== Helper functions (pure) ==========

def _ts(dt):
    """Convert to 'YYYY-MM-DD' string, return None if invalid."""
    normalized = pd.to_datetime(dt, errors="coerce")
    return normalized.strftime("%Y-%m-%d") if pd.notna(normalized) else None


def _basic_clean(name: str) -> str:
    if not name or not isinstance(name, str):
        return ""
    name = name.lower().strip()
    for sw in ["the ", "news", "media", ".com"]:
        name = name.replace(sw, "")
    return re.sub(r"[^a-z0-9]", "", name)


def _publisher_alias(name: str) -> str:
    return PUBLISHER_ALIAS.get(name, name)


def normalize_publisher(name: str) -> str:
    clean = _basic_clean(name)
    return _publisher_alias(clean)


def get_publisher_score(raw):
    key = normalize_publisher(raw)
    return PUBLISHER_WEIGHTS.get(key, DEFAULT_UNKNOWN_SCORE)


def clean_company_name(name: str) -> str:
    remove_terms = [", inc.", "inc.", "corporation", "corp.", "company",
                    "co.", "ltd.", "limited"]
    name = name.lower()
    for term in remove_terms:
        name = name.replace(term, "")
    return name.strip().title()


def fingerprint_news(url: str, headline: str) -> str:
    """Return a stable fingerprint for deduping news articles."""
    return hashlib.md5((url + "|" + headline).strip().lower().encode("utf-8")).hexdigest()


def _scalar(x):
    # Convert a pd.Series with single value to scalar
    if isinstance(x, pd.Series):
        return x.iloc[0]
    return x


# time weights for time_proximity
TIME_WEIGHTS = {0: 1.00, -1: 0.70, 1: 0.50, -2: 0.40}
