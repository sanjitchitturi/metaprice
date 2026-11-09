# Data card

## Bundled synthetic fixture

`examples/synthetic/transactions.csv` is a small synthetic product-demand fixture for
testing the pipeline. It is not representative of commercial demand, does not identify
causal price effects, and must not be used for business-performance claims.

## External data

You must obtain external datasets from their source and comply with their terms. Record
the source URL, retrieval date, license, file checksum, and any cleaning decisions in
your experiment record. MetaPrice stores the input checksum in `run-manifest.json`; it
does not distribute or validate third-party data licenses for you.

## Expected fields

Input data needs `StockCode`, `Quantity`, and `Price` or the supported `UnitPrice`
alias. `Invoice` supports cancellation filtering and `Description` supports the
heuristic category assignment. Inspect preprocessing before interpreting either field
as ground truth.

## Known limitations

The current protocol interpolates observed demand curves and normalizes each product
before splitting support and query points. It is a method-development protocol, not a
temporal or causal evaluation of unseen transactions.
