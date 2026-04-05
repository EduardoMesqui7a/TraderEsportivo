# TraderEsportivo

Quant-Bet Under 2.5 built from the `Trader2` workspace.

## Run

```bash
streamlit run app.py
```

## Notes

- Historical CSVs live in `data/football-data/` locally and are ignored from git.
- If the app runs in Streamlit Cloud without the local CSV folder, it will try to bootstrap the official Football-Data.co.uk CSVs into a writable cache automatically.
- The main app entrypoint is `app.py`.
