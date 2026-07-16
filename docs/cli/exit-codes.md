# CLI exit codes

`vgreens-ai` uses stable exit codes so scripts can react reliably.

| Code | Meaning                          | Typical cause / fix |
|------|----------------------------------|---------------------|
| 0    | success                          | — |
| 1    | generic error                    | see the message on stderr |
| 2    | usage error                      | bad flags/args (Typer) — check `--help` |
| 3    | network / unreachable            | wrong URL, VPN down, timeout |
| 4    | authentication / permission      | bad/expired token, or role not allowed — `vgreens-ai config init` |
| 5    | not found                        | job/model/version doesn't exist |
| 6    | conflict                         | e.g. promoting a non-validated version |
| 7    | server error                     | API returned 5xx — check the server / runbooks |

Human messages and spinners go to **stderr**; machine output (`--output json|yaml`)
goes to **stdout** only, so `... -o json` is safe to pipe.
