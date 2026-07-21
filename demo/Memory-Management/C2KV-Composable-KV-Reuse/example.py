from c2kv_demo import evaluate, format_report, make_documents


def main() -> None:
    documents = make_documents()
    result = evaluate(documents, block_size=4)
    print(format_report(result))


if __name__ == "__main__":
    main()
