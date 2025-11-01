import os

print("1️⃣ Ingest PDFs")
print("2️⃣ Chat with City Info RAG")

choice = input("Enter your choice: ")

if choice == "1":
    os.system("python ingest.py")
elif choice == "2":
    os.system("python chat.py")
else:
    print("Invalid option.")
