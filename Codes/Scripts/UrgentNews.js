var articles = [];
var apiUrl = 'https://api.openai.com/v1/embeddings';
var empty = true;

function extractFoxNews2() {
  const rssUrl = 'https://feeds.foxnews.com/foxnews/latest';
  const response = UrlFetchApp.fetch(rssUrl);
  const xmlData = response.getContentText();
  const document = XmlService.parse(xmlData);
  const rootElement = document.getRootElement();
  const channelElement = rootElement.getChild('channel');
  const itemElements = channelElement.getChildren('item');
  const newsData = [];
  
  const currentDate = new Date();
  const hourAgo = new Date(currentDate - 2 * 60 * 60 * 1000); // Calculate the date one hour ago

  for (const itemElement of itemElements) {
    const title = itemElement.getChildText('title');
    const contentEncodedElement = itemElement.getChild('content:encoded');
    let description;
    
    if (contentEncodedElement) {
      description = contentEncodedElement.getValue().substring(0, 500);
    } else {
      description = itemElement.getChildText('description');
      description = description ? description.substring(0, 500) : 'No description available.';
    }
    
    const link = itemElement.getChildText('guid');
    const pubDate = new Date(itemElement.getChildText('pubDate')); // Convert pubDate to a Date object
    if (pubDate >= hourAgo) {
      addArticle(title, description, link, "Fox News");
    }
  }
}

function extractArticlesFromRSSHub2(source, rssHubUrl, retryCount = 0) {
  const api_Url = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssHubUrl)}`;
  const options = {
    muteHttpExceptions: true,
  };

  try {
    const response = UrlFetchApp.fetch(api_Url, options);
    const data = response.getContentText();
    const jsonData = JSON.parse(data);

    if (!jsonData.items || !Array.isArray(jsonData.items)) {
      // Retry fetching the data up to 3 times
      if (retryCount < 3) {
        return extractArticlesFromRSSHub2(source, rssHubUrl, retryCount + 1);
      } else {
        if (source == 'Fox News') return extractFoxNews2()
      }
    }

    const currentDate = new Date();
    const hourAgo = new Date(currentDate - 2 * 60 * 60 * 1000); // Calculate the date one hour ago
    for (const item of jsonData.items) {
      try {
        const title = item.title;
        const description = item.description;
        const link = item.link;
        const pubDate = new Date(item.pubDate);
        if (pubDate >= hourAgo) {
          addArticle(title, description, link, source);
        }
      } catch (itemError) {
        // You can choose to skip this item by continuing to the next iteration
        continue;
      }
    }
  } catch (error) {
    console.error("Error occurred:", error);
    if (retryCount < 3) {
      return extractArticlesFromRSSHub2(source, rssHubUrl, retryCount + 1);
    } else {
      console.error("Max retries reached. Error: An exception occurred while fetching or processing RSS data.", error);
      return "An error occurred while fetching or processing RSS data.";
    }
  }
}

function addArticle(title, description, link, source) {
  var newArticle = {
    title: title,
    description: description,
    link: link,
    source: source,
    embedding: getOpenAIEmbedding(title + " " + description)
  };
  articles.push(newArticle);
}

async function generateTitlesForUser2(source, news_text, link, let_count = 0, let_count2 = 0) {
  conversationPrompt = `Subject: Request for Generating Editorial News Title.
   Using the provided article's title, description, and link,(${news_text}) generate captivating title, editorial title from its original title and its description. Do not use the same exact words, be creative and make the generated title attractive, and in urgent format.

    Finally, present the newly generated headline alongside its respective link. Please format it as follows:
    - The Generated Headline - <a href="${link}">${source}</a>
    `;
  try {
    var api_response = await send_request(conversationPrompt);
    // Extract the first title and link using regular expressions
    if (!api_response) {
      if (let_count <= 6) {
        return generateTitlesForUser2(source, news_text, link, let_count + 1, let_count2);
      } else {
        console.error("Error: No titles and links found in API response or Empty API response.")
        return "Error: No titles and links found in API response or Empty API response.";
      }
    }
    return api_response;
  } catch (error) {
    if (let_count2 <= 6) {
      return generateTitlesForUser2(source, news_text, link, let_count, let_count2 + 1);
    } else {
      console.error("Error: An exception occurred while generating titles.", error);
      return "No titles were found or generated.";
    }
  }
}

function getOpenAIEmbedding(text) {
  var headers = {
    'Authorization': 'Bearer ' + apiKey,
    'Content-Type': 'application/json'
  };
  var data = {
    'input': text,
    'model': model
  };
  var options = {
    'method': 'post',
    'headers': headers,
    'payload': JSON.stringify(data),
    'muteHttpExceptions': true  // Add this option

  };
  var response = UrlFetchApp.fetch(apiUrl, options);
  var result = JSON.parse(response.getContentText());
  return result['data'][0]['embedding'];
}

function calculateCombinedSimilarity(embedding1, embedding2) {
  if (embedding1.length === 0 || embedding2.length === 0) {
    return 0; // You might want to handle this case differently based on your requirements.
  }

  let dotProduct = 0;
  let magnitude1 = 0;
  let magnitude2 = 0;

  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    magnitude1 += embedding1[i] * embedding1[i];
    magnitude2 += embedding2[i] * embedding2[i];
  }

  // Calculate the magnitudes (Euclidean norms)
  magnitude1 = Math.sqrt(magnitude1);
  magnitude2 = Math.sqrt(magnitude2);

  // Check for zero magnitudes
  if (magnitude1 === 0 || magnitude2 === 0) {
    return 0; // You might want to handle this case differently based on your requirements.
  }

  // Calculate the cosine similarity
  const cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

  // Calculate the Jaccard similarity
  const intersection = embedding1.filter((value, i) => value === 1 && embedding2[i] === 1).length;
  const union = embedding1.filter(value => value === 1).length + embedding2.filter(value => value === 1).length - intersection;
  const jaccardSimilarity = intersection / union;

  // Combine the similarities (adjust weights as needed)
  const combinedSimilarity = (cosineSimilarity + jaccardSimilarity) / 2;

  return combinedSimilarity;
}

function calculateCosineSimilarity(embedding1, embedding2) {
  // Calculate the dot product of embedding1 and embedding2
  let dotProduct = 0;
  let magnitude1 = 0;
  let magnitude2 = 0;

  for (let i = 0; i < embedding1.length; i++) {
    dotProduct += embedding1[i] * embedding2[i];
    magnitude1 += embedding1[i] * embedding1[i];
    magnitude2 += embedding2[i] * embedding2[i];
  }

  // Calculate the magnitudes (Euclidean norms)
  magnitude1 = Math.sqrt(magnitude1);
  magnitude2 = Math.sqrt(magnitude2);

  // Calculate the cosine similarity
  const similarity = dotProduct / (magnitude1 * magnitude2);

  return similarity;
}

function calculateAverageCosineSimilarity(title, embedding) {
  let totalSimilarity = 0;

  for (const article of articles) {
    // Exclude the article with the specified title
    if (article.title !== title) {
      // Calculate similarity for each article (assuming calculateCombinedSimilarity is defined)
      const similarity = calculateCosineSimilarity(embedding, article.embedding);
      totalSimilarity += similarity;
    }
  }

  const averageSimilarity = totalSimilarity / (articles.length - 1); // Exclude the current article
  return averageSimilarity;
}


function findMostSimilarArticle() {
  let mostSimilarArticle = null;
  let highestAverageSimilarity = 0.5;

  for (const article of articles) {
    const averageSimilarity = calculateAverageCosineSimilarity(article.title, article.embedding);
    if (averageSimilarity > highestAverageSimilarity) {
      empty = false;
      highestAverageSimilarity = averageSimilarity;
      mostSimilarArticle = article;
    }
  }

  if(mostSimilarArticle != null){
  console.log("mostSimilarArticle: " + mostSimilarArticle.title + mostSimilarArticle.description);

  }
  return mostSimilarArticle;
}

function findTopFourSimilarArticles() {
  // Find the most similar article
  const mostSimilarArticle = findMostSimilarArticle();
  if (empty === true) {
    return [];
  }

  let sources = [];
  sources.push(mostSimilarArticle.source);

  // Calculate the cosine similarity between the most similar article and all articles
  const similarArticles = articles.map(article => {
  const similarity = calculateCosineSimilarity(mostSimilarArticle.embedding, article.embedding);

    return { article, similarity };
  });

  // Sort the articles by similarity in descending order
  similarArticles.sort((a, b) => b.similarity - a.similarity);

  // Create a Set to keep track of unique articles
  const uniqueArticles = new Set();

  // Return the most similar article at the top, and the next four similar articles
  const similarityThreshold = 0.5;

  const topSimilarArticles = similarArticles
    .filter(item => {
      // Filter out articles with similarity below the threshold
      return item.similarity > similarityThreshold;
    })
    .filter(item => {
      // Filter out the most similar article itself
      if (item.article === mostSimilarArticle) {
        return false;
      }
      if (sources.includes(item.article.source)){
        return false;
      }
      // Use the article's title as a key to check uniqueness
      const key = item.article.title;
      if (!uniqueArticles.has(key)) {
        uniqueArticles.add(key);
        sources.push(item.article.source);
        return true;
      }
      return false;
    })
    .slice(0, 4)
    .map(item => ({
      title: item.article.title,
      description: item.article.description,
      link: item.article.link,
      source: item.article.source,
      similarity: item.similarity,
    }));

  // Add the most similar article at the beginning
  const result = [
    {
      title: mostSimilarArticle.title,
      description: mostSimilarArticle.description,
      link: mostSimilarArticle.link,
      source: mostSimilarArticle.source,
    },
    ...topSimilarArticles.map(item => ({
      title: item.title,
      description: item.description,
      link: item.link,
      source: item.source,
    })),
  ];
  for(const a of result){
    console.log(a.title + a.description);
  }
  if(result.length < 3){
    return "";
  }
  return result;
}

async function sendemail2(mailSubject, mailContent, recepientsMailList, user) {
  var body = `
    Dear ${user},<br><br>
    In the past hour, we've received critical news that we believe you should know about. Please take a moment to read:<br><br>
  `;

  body += mailContent;
  body += `<br><br>Your safety and awareness are our top priorities. If you have questions, please contact our support team.<br><br>
      Thank you for trusting News Wave.<br>
      Sincerely,<br>
      News Wave
    `;
  
  const options = {
    htmlBody: body // Use htmlBody to send the email in HTML format
  };

  // Send the email with the provided content
  GmailApp.sendEmail(recepientsMailList, mailSubject," ", options);
}

async function sendUrgentNews(){
  try {
    // Fetch articles from multiple RSS sources
  const rssSources = [
    { source: 'Fox News', url: 'https://feeds.foxnews.com/foxnews/latest' },
    { source: 'The New York Times', url: 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml' },
    { source: 'The Guardian', url: 'https://www.theguardian.com/world/rss' },
    { source: 'Al Jazeera', url: 'https://www.aljazeera.com/xml/rss/all.xml' },
    { source: 'BBC News', url: 'http://feeds.bbci.co.uk/news/rss.xml' },
    { source: 'Time', url: 'https://time.com/feed/' },
  ];

  for (const sourceInfo of rssSources) {
    extractArticlesFromRSSHub2(sourceInfo.source, sourceInfo.url);
  }

  // Find the most similar articles
  const topSimilarArticles = findTopFourSimilarArticles();

    // Check if there are any similar articles to send
  if (topSimilarArticles.length == 0) {
    console.warn('No similar articles found.');
    return;
  }

  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  const mailSubject = "Urgent News Alert: Important Updates in the Last Hour";

  const columnNumber = 1;
  const columnData = sheet.getRange(1, columnNumber, sheet.getLastRow(), 1).getValues();
  const rowCount = columnData.filter(value => value[0] !== '').length;

  const responsePromises = [];
  for (const list of topSimilarArticles) {
    var feed = 'title: ' + list.title + ' ' + 'description: ' + list.description;
    responsePromises.push(generateTitlesForUser2(list.source, feed, list.link));
  }

  const responses = await Promise.all(responsePromises);
  // Loop through each row of user data
  for (let i = 2; i < rowCount + 2; i++) {
    const range = sheet.getRange("C" + i);
    const valueEmail = range.getValue();
    const range2 = sheet.getRange("F" + i);
    const valueName = range2.getValue();
    var j = 1;
    if (responses.length > 0) {
      sendemail2(mailSubject, responses.join('<br>\n'), valueEmail, valueName);
    }
  }  
  } catch (error) {
    console.error("Error occurred in sendUrgentNews:", error);
  }
}