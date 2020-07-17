from django.shortcuts import render
from .predict import predict

def sentiment_extraction_view(request):
    if request.method == 'GET':
        return render(request, "sentiment_extraction/form.html")
    elif request.method == 'POST':
        print(request.POST)
        tweet = request.POST["content_text"]
        sentiment = request.POST["sentiment"]
        selected_text = predict(tweet, sentiment)
        context = {
            "sentiment": sentiment,
            "tweet": tweet,
            "selected_text": selected_text
        }
        return render(request, "sentiment_extraction/form.html", context)
