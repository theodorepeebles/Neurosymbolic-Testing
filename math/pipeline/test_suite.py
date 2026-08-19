# ── MATH WORD PROBLEM BANK ─────────────────────────────────────────────────────────

test_suite = [
    # {
    #     "problem": "A warehouse has 3 shelves. Each shelf holds 144 items. How many items total?",
    #     "answer": 432
    # },
    # {
    #     "problem": "A road crew paves 278 meters per day for 365 days. What is the total meters paved?",
    #     "answer": 101470
    # },
    # {
    #     "problem": "A store had 10000 units in stock. They sold 3742. How many units remain?",
    #     "answer": 6258
    # },
    {
        "problem": "Two shipments arrive at a warehouse: 45678 boxes and 23456 boxes. What is the combined total?",
        "answer": 69134
    },
    # {
    #     "problem": "A server processes 9999999 requests per hour for 8761 hours. What is the total number of requests?",
    #     "answer": 87609991239
    #  },
    # {
    #     "problem": "A printer produces 123456 pages per minute over 98765 minutes. How many pages total?",
    #     "answer": 12193131840
    # },
    # {
    #     "problem": "Split 100 items among 4 workers equally. How many does each worker get?",
    #     "answer": 25
    # },
    # {
    #     "problem": "A factory has 8400 parts to pack into boxes of 24. How many boxes are needed?",
    #     "answer": 350
    # },
    # {
    #     "problem": "A 360-kilometer road is divided into 9 equal sections. How long is each section?",
    #     "answer": 40
    # },
    # {
    #     "problem": "A worker earns $18 per hour and works 8 hours a day for 5 days. What is the total pay?",
    #     "answer": 720
    # },
    # {
    #     "problem": "A box holds 12 bottles. A crate holds 8 boxes. There are 6 crates. How many bottles total?",
    #     "answer": 576
    # },
    # {
    #     "problem": "A car travels at 60 km/h for 3 hours, then 80 km/h for 2 hours. What is the total distance?",
    #     "answer": 340
    # },
    # {
    #     "problem": "If you multiply your monthly salary of $4500 by 12 months, what is the annual total?",
    #     "answer": 54000
    # },
    # {
    #     "problem": "A tank holds 500 liters. You have 200 such tanks. What is the total capacity?",
    #     "answer": 100000
    # },
    # {
    #     "problem": "Each class has 30 students. There are 12 classes in the school. What is the total student count?",
    #     "answer": 360
    # },
    # {
    #     "problem": "A factory produces 0 units per day for 99999 days. How many units are produced?",
    #     "answer": 0
    # },
    # {
    #     "problem": "15 groups of 15 people attend an event. How many people total?",
    #     "answer": 225
    # },
    # {
    #     "problem": "A runner completes 1 lap per minute for 1 minute. How many laps does the runner complete?",
    #     "answer": 1
    # },
    # {
    #     "problem": "A library had 2400 books. 318 were damaged and removed. How many books remain?",
    #     "answer": 2082
    # },
    # {
    #     "problem": "A project needs 500 hours of work. 213 hours have been completed. How many hours remain?",
    #     "answer": 287
    # },
    # {
    #     "problem": "If 7 trucks each carry 3500 kg of gravel, what is the total weight of gravel delivered?",
    #     "answer": 24500
    # },
    # {
    #     "problem": "A recipe calls for 250 grams of flour per batch. How much flour is needed for 16 batches?",
    #     "answer": 4000
    # },

    # # ==============================
    # # HARD NEURO-SYMBOLIC STRESS TESTS, 4-7 step, large numbers, mixing operations
    # # ==============================

    # {
    #     "problem": "A data center processes 12,345,678,901 requests per second for 98,765 seconds. How many requests are processed in total?",
    #     "answer": 1219320976657265
    # },
    # {
    #     "problem": "A factory produces 9,876,543 units per day for 123,456 days. How many units are produced?",
    #     "answer": 1219318492608
    # },
    # {
    #     "problem": "A telescope records 8,888,888,888 observations per hour for 77,777 hours. How many observations are recorded?",
    #     "answer": 691351111041976
    # },

    # {
    #     "problem": "A warehouse contains 123,456 pallets. Each pallet holds 789 boxes. Each box contains 45 items. How many items are stored in total?",
    #     "answer": 4383305280
    # },
    # {
    #     "problem": "A shipping company operates 4,321 trucks. Each truck carries 987 crates. Each crate contains 654 products. How many products are being transported?",
    #     "answer": 2789196858
    # },
    # {
    #     "problem": "A university has 2,468 classrooms. Each classroom contains 135 desks. Each desk seats 2 students. How many students can be seated?",
    #     "answer": 666360
    # },

    # {
    #     "problem": "A worker earns $37 per hour, works 11 hours per day, 6 days per week, for 52 weeks. What is the total annual pay?",
    #     "answer": 126984
    # },
    # {
    #     "problem": "A solar farm generates 4,567 kWh per hour. It operates 24 hours per day for 365 days. How much energy is produced annually?",
    #     "answer": 40006920
    # },
    # {
    #     "problem": "A factory makes 789 parts per machine-hour. There are 456 machines operating 18 hours per day for 365 days. How many parts are produced?",
    #     "answer": 2363780880
    # },

    # {
    #     "problem": "A company has 123 offices. Each office has 456 employees. Each employee works 230 days per year and processes 78 forms per day. How many forms are processed annually?",
    #     "answer": 1006218720
    # },
    # {
    #     "problem": "A library system has 89 branches. Each branch has 234 shelves. Each shelf contains 567 books. Each book is borrowed 12 times per year. How many annual borrowings occur?",
    #     "answer": 141700104
    # },

    # {
    #     "problem": "A logistics network has 67 hubs. Each hub serves 234 warehouses. Each warehouse contains 789 aisles. Each aisle stores 456 bins. Each bin contains 123 items. How many items are stored in the network?",
    #     "answer": 693805306896
    # },
    # {
    #     "problem": "A cloud provider operates 432 data centers. Each center contains 876 racks. Each rack contains 45 servers. Each server handles 12,345 requests per second for 86,400 seconds. How many requests are processed in a day?",
    #     "answer": 18163736939520000
    # },

    # {
    #     "problem": "A manufacturer owns 54 factories. Each factory has 123 production lines. Each line has 45 machines. Each machine produces 678 units per hour. The machines run 16 hours per day for 312 days per year. How many units are produced annually?",
    #     "answer": 1011615920640
    # },
    # {
    #     "problem": "A retailer has 321 stores. Each store has 98 departments. Each department contains 76 shelves. Each shelf holds 54 products. Each product sells 32 times per month for 12 months. How many annual sales occur?",
    #     "answer": 49575794688
    # },

    # {
    #     "problem": "A nation has 1,234 cities. Each city contains 567 schools. Each school has 89 classrooms. Each classroom contains 34 students. Each student completes 12 assignments per course. They take 8 courses per year and attend for 4 years. How many assignments are completed?",
    #     "answer": 813014641152
    # },
    # {
    #     "problem": "A computing cluster contains 345 facilities. Each facility has 678 rooms. Each room has 90 racks. Each rack contains 40 servers. Each server runs 64 cores. Each core executes 2,500,000 operations per second for 31,536,000 seconds. How many operations are executed in one year?",
    #     "answer": 4248913397760000000000000
    # },

    # {
    #     "problem": "A company starts with 9,876,543,210 units in inventory. It receives 7 shipments of 1,234,567,890 units each and then sells 4,567,890,123 units. How many units remain?",
    #     "answer": 13950628317
    # },
    # {
    #     "problem": "A machine produces 123,456 units per hour for 456 hours. Of those units, 17 percent fail inspection. How many pass inspection?",
    #     "answer": 46725626
    # },
    # {
    #     "problem": "A warehouse contains 987 pallets. Each pallet has 654 boxes. Each box contains 321 items. If 23 percent of the items are shipped out, how many remain?",
    #     "answer": 159547740
    # },
    # {
    #     "problem": "A fund starts with $1,234,567,890. It gains $98,765,432 per month for 36 months, then loses $876,543,210. What is the final balance?",
    #     "answer": 3913580232
    # },
    # {
    #     "problem": "A network has 456 servers. Each server handles 789 requests per second. The network runs continuously for 365 days. If 2 percent of requests fail, how many successful requests occur?",
    #     "answer": 11119225259520
    # }
]