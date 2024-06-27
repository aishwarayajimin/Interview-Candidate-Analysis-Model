import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from wordcloud import WordCloud
import pandas as pd
from collections import defaultdict


def generate_reports():
    # Load the JSON data
    with open('transcribed_text.json', 'r', encoding='utf-8') as f:
        transcribed_text = json.load(f)

    with open('sentiment_scores.json', 'r', encoding='utf-8') as f:
        sentiment_scores = json.load(f)

    with open('emotion_data.json', 'r', encoding='utf-8') as f:
        emotions_over_time = json.load(f)
    
    with open('evaluation_results.json', 'r', encoding='utf-8') as f:
        evaluation_results = json.load(f)
    with open('audio_analysis_results.json', 'r', encoding='utf-8') as f:
        audio_analysis_results = json.load(f)
    with open('personality_prediction_result.json', 'r',encoding='utf-8') as f:
        personality_prediction_result=json.load(f)
    with open('emotions.json', 'r', encoding='utf-8') as f:
        emotion_aggregate = json.load(f)


    # Generate the emotion chart
    def generate_emotion_chart(emotions_over_time):
        if emotions_over_time:
            emotion_data = {emotion: [] for emotion in emotions_over_time[0]}
            for emotion_dict in emotions_over_time:
                for emotion, score in emotion_dict.items():
                    emotion_data[emotion].append(score)

            plt.figure(figsize=(12, 6))
            for emotion, scores in emotion_data.items():
                plt.plot(scores, label=emotion)
            plt.xlabel('Frame')
            plt.ylabel('Emotion Score')
            plt.title('Emotions Over Time')
            plt.legend()

            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close()
            buf.seek(0)
            return base64.b64encode(buf.read()).decode('utf-8')
        else:
            return None
    def generate_emotion_pie_chart(emotion_aggregate):
        emotion_aggregate = defaultdict(float, emotion_aggregate)

        # Normalize the emotion scores
        total_emotions = sum(emotion_aggregate.values())
        for emotion in emotion_aggregate:
            emotion_aggregate[emotion] /= total_emotions

        labels = list(emotion_aggregate.keys())
        sizes = list(emotion_aggregate.values())

        # Generate the pie chart
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')
        plt.title('Emotion Distribution')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        pie_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
        insights = generate_insights(emotion_aggregate)
        
        return pie_chart_base64, insights

# Generate the pie chart
      
    # Generate the emotion distribution bar chart
    def generate_emotion_bar_chart(emotion_aggregate):
        emotion_aggregate = defaultdict(float, emotion_aggregate)

    # Normalize the emotion scores
        total_emotions = sum(emotion_aggregate.values())
        for emotion in emotion_aggregate:
            emotion_aggregate[emotion] /= total_emotions

        labels = list(emotion_aggregate.keys())
        sizes = list(emotion_aggregate.values())

        # Generate the bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color=['blue', 'green', 'red', 'purple', 'orange', 'cyan'])
        plt.xlabel('Emotions')
        plt.ylabel('Percentage')
        plt.title('Emotion Distribution')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)

        bar_chart_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
        insights = generate_insights(emotion_aggregate)
        
        return bar_chart_base64, insights
    def generate_insights(emotion_aggregate):
        insights = []

        # Example insights, you can customize this based on your specific analysis
        if 'happy' in emotion_aggregate and 'sad' in emotion_aggregate:
            if emotion_aggregate['happy'] > emotion_aggregate['sad']:
                insights.append("The happiness level seems to be higher compared to sadness.")
            else:
                insights.append("Sadness is more prevalent compared to happiness.")
        
        if 'surprised' in emotion_aggregate and 'angry' in emotion_aggregate:
            if emotion_aggregate['surprised'] > emotion_aggregate['angry']:
                insights.append("There are more instances of surprise than anger.")
            else:
                insights.append("Anger is more frequently detected compared to surprise.")

        return insights

        # Generate the charts
    


    # Generate the sentiment analysis chart
    def generate_sentiment_chart(sentiment_scores):
        labels = ['neg', 'neu', 'pos']
        sizes = [sentiment_scores[label] for label in labels]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title('Sentiment Analysis')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')

    # Generate the sentiment analysis bar chart
    def generate_sentiment_bar_chart(sentiment_scores):
        labels = ['neg', 'neu', 'pos']
        sizes = [sentiment_scores[label] for label in labels]

        plt.figure(figsize=(10, 6))
        plt.bar(labels, sizes, color=['red', 'blue', 'green'])
        plt.xlabel('Sentiment')
        plt.ylabel('Percentage')
        plt.title('Sentiment Analysis')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    
 
    

    # Generate the competency occurrences chart
    # Generate the combined competency graph
    def generate_combined_competency_graph(evaluation_results):
       competencies = ['Leadership', 'communication', 'positive_attitude', 'teamwork']
       metrics = ['precision', 'recall', 'f1-score', 'support']
       combined_data = {metric: [] for metric in metrics}

       for competency in competencies:
          data = evaluation_results['classification_report'][competency]
          for metric in metrics:
            combined_data[metric].append(data[metric])
       x = range(len(competencies))
       width = 0.2
       plt.figure(figsize=(14, 7))
       colors = ['Magenta', 'Lime', 'Indigo', 'Coral']
    
       for i, metric in enumerate(metrics):
           plt.bar([p + width * i for p in x], combined_data[metric], width=width, label=metric, color=colors[i])

       plt.xlabel('Competencies')
       plt.ylabel('Scores')
       plt.title('Comparison of Competencies')
       plt.xticks([p + 1.5 * width for p in x], competencies)
       plt.legend()
    

    # Add annotations for clarity
       for i, metric in enumerate(metrics):
           for j, value in enumerate(combined_data[metric]):
             plt.text(j + width * i, value + 0.1, f'{value:.2f}', ha='center', va='bottom')

       buf = BytesIO()
       plt.savefig(buf, format='png', bbox_inches='tight')
       plt.close()
       buf.seek(0)
       graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Generate detailed insights based on the data
       insights = "<b>Overall Insights:</b><br>"
       overall_precision = sum(combined_data['precision']) / len(competencies)
       overall_recall = sum(combined_data['recall']) / len(competencies)
       overall_f1_score = sum(combined_data['f1-score']) / len(competencies)
     
       insights += (
           f"Overall Precision: {overall_precision:.2f} - Indicates the average precision across all competencies.<br>"
           f"Overall Recall: {overall_recall:.2f} - Indicates the average recall across all competencies.<br>"
           f"Overall F1-Score: {overall_f1_score:.2f} - Indicates the average F1-score across all competencies.<br><br>"
    )
    
       
    
       return graph_base64, insights
    def generate_word_cloud(transcript):
        transcript = transcribed_text.get('text', '')
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(transcript)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')

        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return base64.b64encode(buf.read()).decode('utf-8')
    word_cloud_image = generate_word_cloud(transcribed_text)

    # Generate the emotion chart image
    emotion_chart_image = generate_emotion_chart(emotions_over_time)
    sentiment_chart_image = generate_sentiment_chart(sentiment_scores)
    sentiment_bar_chart_image = generate_sentiment_bar_chart(sentiment_scores)
    pie_chart_base64,pie_chart_insights = generate_emotion_pie_chart(emotion_aggregate)

    bar_chart_base64 ,bar_chart_insights= generate_emotion_bar_chart(emotion_aggregate)
    # transcript_with_timestamps_image = generate_competency_chart(transcript_with_timestamps)
   # competency_graphs = generate_competency_graphs(evaluation_results)
   # combined_competency_graph_image = generate_combined_competency_graph(evaluation_results)
    graph_base64, insights_html = generate_combined_competency_graph(evaluation_results)
    accuracy = evaluation_results['accuracy']
    
    classification_report = evaluation_results['classification_report']
    # Create HTML table from classification report
    classification_report_html = """
    <table>
        <thead>
            <tr>
                <th>Category</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Support</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for category, metrics in classification_report.items():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            classification_report_html += f"""
            <tr>
                <td>{category}</td>
                <td>{metrics['precision']:.2f}</td>
                <td>{metrics['recall']:.2f}</td>
                <td>{metrics['f1-score']:.2f}</td>
                <td>{metrics['support']}</td>
            </tr>
            """
    
    # Add summary rows
    classification_report_html += f"""
            <tr>
                <td>Macro Avg</td>
                <td>{classification_report['macro avg']['precision']:.2f}</td>
                <td>{classification_report['macro avg']['recall']:.2f}</td>
                <td>{classification_report['macro avg']['f1-score']:.2f}</td>
                <td>{classification_report['macro avg']['support']}</td>
            </tr>
            <tr>
                <td>Weighted Avg</td>
                <td>{classification_report['weighted avg']['precision']:.2f}</td>
                <td>{classification_report['weighted avg']['recall']:.2f}</td>
                <td>{classification_report['weighted avg']['f1-score']:.2f}</td>
                <td>{classification_report['weighted avg']['support']}</td>
            </tr>
        </tbody>
    </table>
    """
    audio_analysis_html = f"""
    <h2>Audio Analysis Results</h2>
   <table>
    <thead>
        <tr>
            <th>Metric</th>
            <th>Value</th>
            <th>Standard</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Pitch</td>
            <td>{audio_analysis_results['pitch']:.2f} Hz</td>
            <td>100-300 Hz</td>
        </tr>
        
        <tr>
            <td>Speech Rate</td>
            <td>{audio_analysis_results['speech_rate']:.2f} words/sec</td>
            <td>2.33-3.17 words/sec (140-190 wpm)</td>
        </tr>
        <tr>
            <td>Pauses</td>
            <td>{audio_analysis_results['pauses_per_minute']:.2f} per/sec</td>
            <td>6-10 per min (short: 0.15-0.75 sec, long: 1-2 sec)</td>
        </tr>
    </tbody>
</table>

    """
    personality_prediction_html = f"""
    <h3>Personality Type:</h3>
    <p>Predicted Personality: {personality_prediction_result['predicted_personality']}</p>
   
"""

    candidate_decision_html = f"""
   
"""
    personality_df = pd.read_csv('mbti_personality_info.csv')
    personality_type = personality_prediction_result['predicted_personality']
    personality_type_accuracy = personality_prediction_result['predicted_personality_accuracy']
    personality_description = personality_df[personality_df['Type'] == personality_type]['Description'].values[0]
    personality_link= personality_df[personality_df['Type'] == personality_type]['Link'].values[0]

    # Create the HTML report
    html_report = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Video and Audio Analysis Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: #f0f0f0;
                margin: 0;
                padding: 0;
            }}
            .sidebar {{
                height: 100%;
                width: 250px;
                position: fixed;
                top: 0;
                left: 0;
                background-color: #333;
                padding-top: 20px;
            }}
            .sidebar a {{
                padding: 10px 15px;
                text-decoration: none;
                font-size: 18px;
                color: white;
                display: block;
            }}
            .sidebar a:hover {{
                background-color: #575757;
            }}
            .content {{
                margin-left: 260px;
                padding: 20px;
            }}
            .header {{
                background: #333;
                color: white;
                padding: 10px 20px;
                text-align: center;
            }}
            .card {{
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }}
            .card h2 {{
                margin: 0 0 10px;
            }}
            .card img {{
                max-width: 100%;
                height: auto;
            }}
            .button {{
                display: inline-block;
                padding: 10px 20px;
                margin: 10px;
                font-size: 16px;
                cursor: pointer;
                text-align: center;
                text-decoration: none;
                outline: none;
                color: #fff;
                background-color: #4CAF50;
                border: none;
                border-radius: 15px;
                box-shadow: 0 9px #999;
            }}
            .button:hover {{
                background-color: #3e8e41;
            }}
            .button:active {{
                background-color: #3e8e41;
                box-shadow: 0 5px #666;
                transform: translateY(4px);
            }}
            .hidden {{
                display: none;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-top: 20px;
            }}
            th, td {{
                border: 1px solid #000;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #4CAF50;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
        </style>
        <script>
            function toggleVisibility(id) {{
                var element = document.getElementById(id);
                if (element.classList.contains('hidden')) {{
                    element.classList.remove('hidden');
                }} else {{
                    element.classList.add('hidden');
                }}
            }}
           
            
        </script>
    </head>
    <body>
        <div class="sidebar">
                        
            <a href="#emotions">Emotion Analysis</a>
         
            <a href="#sentiments">Sentiment Analysis</a>
            <a href="#transcription">Transcribed Text</a>
            <a href="#scores">Sentiment Scores</a>
            <a href="#evaluation">Model Evaluation</a>
            <a href="#combined_competency_graph">Competency Graphs</a>
            <a href="#insight">Insight</a>
        </div>
        <div class="content">
            <div class="header">
                <h1>Video and Audio Analysis Dashboard</h1>
            </div>
            <div id="emotions" class="card">
                <h2>Emotion Analysis Over Time</h2>
                <button class="button" onclick="toggleVisibility('emotionChart')"> Emotion Chart</button>
                <img id="emotionChart" src="data:image/png;base64,{emotion_chart_image}" alt="Emotion Analysis Over Time" class="hidden"/>
                <p>The x-axis (Frame) indicates the time or sequence of frames, while the y-axis (Emotion Score) indicates the intensity of the emotions from 0 to 1. Each line shows how the intensity of a particular emotion changes over the sequence of frames.</p>

                 <button class="button" onclick="toggleVisibility('emotionPieChart')">Emotion Pie Chart</button>
                <button class="button" onclick="toggleVisibility('emotionBarChart')">Emotion Bar Chart</button>

                  <div id="emotionPieChart" class="hidden">
            <h3>Emotion Pie Chart</h3>
            <img src="data:image/png;base64,{pie_chart_base64}" alt="Emotion Pie Chart">
            <div class="insights">
                <h4>Insights:</h4>
                {''.join(f'<p>{insight}</p>' for insight in pie_chart_insights)}
            </div>
        </div>

        <div id="emotionBarChart" class="hidden">
            <h3>Emotion Bar Chart</h3>
            <img src="data:image/png;base64,{bar_chart_base64}" alt="Emotion Bar Chart">
            <div class="insights">
                <h4>Insights:</h4>
                {''.join(f'<p>{insight}</p>' for insight in bar_chart_insights)}
            </div>
        </div>

            </div>
            

            <div id="sentiments" class="card">
                <h2>Sentiment Analysis</h2>
                <button class="button" onclick="toggleVisibility('sentimentPieChart')"> Sentiment Pie Chart</button>
                <img id="sentimentPieChart" src="data:image/png;base64,{sentiment_chart_image}" alt="Sentiment Analysis" class="hidden"/>
                <button class="button" onclick="toggleVisibility('sentimentBarChart')"> Sentiment Bar Chart</button>
                <img id="sentimentBarChart" src="data:image/png;base64,{sentiment_bar_chart_image}" alt="Sentiment Bar Analysis" class="hidden"/>
            </div>

            <div id="transcription" class="card">
                <h2>Transcribed Text</h2>
                <button class="button" onclick="toggleVisibility('transcribedText')"> Transcribed Text</button>
                <div id="transcribedText" class="hidden">{transcribed_text["text"]}</div>
                <div id="transcribedText" class = "card">{audio_analysis_html}</div>
            </div>

            <div id="scores" class="card">
                <h2>Sentiment Scores</h2>
                <div>
                    <p>Negative: {sentiment_scores["neg"]:.2f}% <span class="emoji">😢</span></p>
                    <p>Neutral: {sentiment_scores["neu"]:.2f}% <span class="emoji">😐</span></p>
                    <p>Positive: {sentiment_scores["pos"]:.2f}% <span class="emoji">😊</span></p>
                    <p>Sentiment Polarity: {sentiment_scores["polarity"]:.2f}</p>
                    <p>The sentiment polarity score indicates the overall emotional tone of the interview text. A score closer to 1 suggests a highly positive sentiment, while a score closer to -1 indicates a highly negative sentiment. A score around 0 suggests neutral sentiment.</p>
                    <p>Sentiment Subjectivity: {sentiment_scores["subjectivity"]:.2f}</p>
                    <p>The sentiment subjectivity score measures the degree of personal opinion, emotion, or feeling expressed in the interview text. A score closer to 1 indicates highly subjective content, where personal feelings and opinions heavily influence the text. A score closer to 0 suggests objective, factual information with minimal personal bias.</p>
                </div>
            </div>

            <div id="evaluation" class="card">
                <h2>Model Evaluation</h2>
                <h3>Accuracy</h3>
                <p>{accuracy:.2f}</p>
                <h3>Classification Report</h3>
                {classification_report_html}
            </div>
           

            <div id="combined_competency_graph" class="card">
    <h2>Combined Competency Graph</h2>
    <button class="button" onclick="toggleVisibility('combined_competency_graph_img')"> Combined Graph</button>
    <img id="combined_competency_graph_img" src="data:image/png;base64,{graph_base64}" alt="Competency Comparison Graph" class="hidden"/>
</div>
            <div id = "insight" class= "card">
                <h2>Insights:</h2>
                {insights_html}
              
               <div id="personality-prediction-html">
                <h2>Personality Prediction</h2>
        <p><b>Predicted Personality Type:</b> {personality_type}</p>
        <b>Predicted Personality Accuracy:</b> {personality_type_accuracy:.2f}</p>
        <p><b>Description:</b> {personality_description}</p>
        <p>If you want to learn more about this particular personality type, you can click <a href="{personality_link}" target="_blank">here</a>.</p>
            </div>
               
            <div id="word_cloud" class="card">
            <h2>Word Cloud</h2>
            <img src="data:image/png;base64,{word_cloud_image}" alt="Word Cloud"></div>
            </div>
        </div>
 
    </body>
    </html>
    """

    # Save the HTML report to a file
    with open('analysis_report.html', 'w', encoding='utf-8') as f:
        f.write(html_report)
    print("HTML report generated successfully!")

if __name__ == "__main__":
    generate_reports()

