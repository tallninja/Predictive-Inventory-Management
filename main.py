import pickle
import gradio as gr

with open("model.pkl", "rb") as f:
    model = pickle.load(f)


def infer(productId, sales, buyingPrice, quantity):
    prediction = model.predict([[productId, sales, buyingPrice, quantity]])
    if prediction <= 0:
        return "You don't need to restock"
    else:
        return "You need to restock ASAP!"


interface = gr.Interface(
    fn=infer, inputs=[gr.Number(), gr.Number(), gr.Number(), gr.Number()], outputs="text")

if __name__ == '__main__':
    interface.launch()
