import duckdb
import argparse

try:
    from tabulate import tabulate
    TABULATE = True
except ImportError:
    TABULATE = False

DB_PATH = "db/uk_jobs.duckdb"

def fetch_jobs(title=None, company=None):
    con = duckdb.connect(DB_PATH)
    query = "SELECT * FROM jobs"
    params = []
    filters = []
    if title:
        filters.append("LOWER(title) LIKE ?")
        params.append(f"%{title.lower()}%")
    if company:
        filters.append("LOWER(company) LIKE ?")
        params.append(f"%{company.lower()}%")
    if filters:
        query += " WHERE " + " AND ".join(filters)
    results = con.execute(query, params).fetchall()
    columns = [desc[0] for desc in con.description]
    con.close()
    return columns, results

def main():
    parser = argparse.ArgumentParser(
        description='''Review UK AI/tech jobs data in DuckDB.

Examples:
  python review_uk_jobs.py
  python review_uk_jobs.py --title Engineer
  python review_uk_jobs.py --company Google
''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--title", help="Filter by job title (case-insensitive, substring match)")
    parser.add_argument("--company", help="Filter by company (case-insensitive, substring match)")
    args = parser.parse_args()

    columns, results = fetch_jobs(args.title, args.company)
    if not results:
        print("No jobs found with the specified criteria.")
        return
    if TABULATE:
        print(tabulate(results, headers=columns, tablefmt="github"))
    else:
        # Fallback: print as plain text
        print(" | ".join(columns))
        print("-" * (len(columns) * 15))
        for row in results:
            print(" | ".join(str(x) for x in row))

if __name__ == "__main__":
    main()