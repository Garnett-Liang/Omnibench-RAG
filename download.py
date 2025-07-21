import arxiv
import os
import random

# Utility: sanitize filename
def sanitize_filename(name):
    return ''.join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in name)

# Search papers using keyword instead of category
def search_papers_by_keyword(keyword):
    print(f"\nSearching for papers with keyword: {keyword}")
    
    search = arxiv.Search(
        query=keyword,
        max_results=30,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    papers = list(search.results())
    if not papers:
        print("No results found.")
        return []
    
    selected = random.sample(papers, min(10, len(papers)))
    return selected

# Show papers list
def show_papers(papers):
    print("\nFound Papers:")
    for idx, paper in enumerate(papers):
        print(f"[{idx}] {paper.title}")

# Download one paper to folder
def download_paper(paper, keyword):
    folder_name = os.path.join("RAG_raw", keyword.replace(" ", "_"))
    os.makedirs(folder_name, exist_ok=True)

    filename = f"{sanitize_filename(paper.title[:100].replace(' ', '_'))}.pdf"
    filepath = os.path.join(folder_name, filename)

    try:
        print(f"⬇ Downloading: {paper.title}")
        paper.download_pdf(dirpath=folder_name, filename=filename)
        print(f"Saved to: {filepath}")
    except Exception as e:
        print(f"Failed to download: {e}")

# Encapsulated main function
def download_rag(keyword):
    keyword = keyword.strip()
    papers = search_papers_by_keyword(keyword)

    if not papers:
        return

    show_papers(papers)

    choice = input("\nDo you want to download all papers? (yes/no): ").strip().lower()
    if choice == 'yes':
        for paper in papers:
            download_paper(paper, keyword)
    else:
        selected = input("Enter the indices of papers you want to download (comma-separated): ")
        try:
            indices = [int(i.strip()) for i in selected.split(',')]
            for i in indices:
                if 0 <= i < len(papers):
                    download_paper(papers[i], keyword)
                else:
                    print(f" Index {i} is out of range.")
        except:
            print("Invalid input.")

# Optional: run as script
if __name__ == "__main__":
    user_input = input("Enter your research topic or domain: ")
    download_rag(user_input)
