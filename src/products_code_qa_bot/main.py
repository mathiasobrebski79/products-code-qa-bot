from transformers import pipeline
import pandas as pd


def main():
    

    code_table = pd.read_csv("data/codes.csv").astype(str)

    # google/tapas-base-finetuned-wtq

    qa_bot = pipeline(
        model="google/tapas-small-finetuned-sqa"
        )
    
    print("Hello from products-code-qa-bot! Type exit to stop")
    
    while True :

        question = input("Ask a question (or type 'exit'): ")

        if question.lower() == "exit":
            break
    
        result = qa_bot(
            query=question, 
            table=code_table
            )
        
        print(f"Answer: {result['answer']}\n")
    


if __name__ == "__main__":
    main()
