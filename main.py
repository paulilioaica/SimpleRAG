from database import Database


def main():
    db = Database("http://localhost:6333")
    db.create_index("my_index", 128)
    db.insert("my_index", ["1", "2"], [[0.1] * 128, [0.2] * 128])
    print(db.search("my_index", [0.1] * 128, 1))
    db.delete("my_index", ["1"])
    db.flush("my_index")
    db.drop_index("my_index")
    print(db.get_index_info("my_index"))

if __name__ == "__main__":
    main()
