var apiKey = "sk-...";
var openAIUrl = 'https://api.openai.com/v1/chat/completions';
//var model = 'gpt-4-0125-preview';

var model = 'text-embedding-ada-002';
var keywordsModel = "ft:gpt-4o-mini-2024-07-18:open-ai-website:datasettest:9xt91rAS"; 

function getKeywords(content) {
    var messages = [
    {"role": "system", "content": "You are a helpful assistant. I will provide you with news article, and you should extract their keywords from them."},
    {"role": "user", "content": content}
  ];
  
  var payload = {
    "model": keywordsModel,
    "messages": messages,
    "max_tokens": 150
  };

  var options = {
    "method": "post",
    "headers": {
      "Authorization": "Bearer " + apiKey,
      "Content-Type": "application/json"
    },
    "payload": JSON.stringify(payload),
    "muteHttpExceptions": true
  };

  var response = UrlFetchApp.fetch(openAIUrl, options);
  var json = JSON.parse(response.getContentText());
  var keywords = json['choices'][0]['message']['content'];
  console.log(content);
  console.log(keywords);
  return keywords;  // Logs the AI's response to the Apps Script console
}

function extractFoxNews() {
    const rssUrl = 'https://feeds.foxnews.com/foxnews/latest';
    const response = UrlFetchApp.fetch(rssUrl);
    const xmlData = response.getContentText();
    const document = XmlService.parse(xmlData);
    const rootElement = document.getRootElement();
    const channelElement = rootElement.getChild('channel');
    const itemElements = channelElement.getChildren('item');
    const newsData = [];

    const currentDate = new Date();
    const twentyFourHoursAgo = new Date(currentDate - 24 * 60 * 60 * 1000); // Calculate the date 24 hours ago

    for (const itemElement of itemElements) {
        const title = itemElement.getChildText('title');

        let description = '';
        // Attempt to extract the full content
        const namespace = XmlService.getNamespace('http://purl.org/rss/1.0/modules/content/');
        const contentEncodedElement = itemElement.getChild('encoded', namespace);

        if (contentEncodedElement) {
            description = contentEncodedElement.getText();
        } else {
            description = itemElement.getChildText('description');
            description = description ? description : 'No description available.';
        }

        // Always clean the description
        description = cleanHtmlContent(description);

        const link = itemElement.getChildText('link'); // Use 'link' instead of 'guid' for the article URL
        const pubDate = new Date(itemElement.getChildText('pubDate')); // Convert pubDate to a Date object

        const keywords = getKeywords(title + " " + description)
        const embeddings = getOpenAIEmbedding(keywords);
        // Check if the article was published in the last 24 hours
        if (pubDate >= twentyFourHoursAgo) {
            // Push the article to the newsData array
            newsData.push({ title, description, link, embeddings });
        }
    }

    // Return the array of articles
    return newsData;
}

function cleanHtmlContent(htmlContent) {
    // Remove <a> tags and their contents
    let plainText = htmlContent.replace(/<a[^>]*>(.*?)<\/a>/g, '$1'); // Keep the text within <a> tags

    // Remove all remaining HTML tags
    plainText = plainText.replace(/<[^>]+>/g, '');

    // Replace HTML entities with plain text equivalents
    plainText = plainText.replace(/&nbsp;/g, ' ')
                         .replace(/&amp;/g, '&')
                         .replace(/&quot;/g, '"')
                         .replace(/&apos;/g, "'")
                         .replace(/&lt;/g, '<')
                         .replace(/&gt;/g, '>');

    // Remove extra spaces and trim leading/trailing spaces
    plainText = plainText.replace(/\s+/g, ' ').trim();

    return plainText;
}

 function extractArticlesFromRSSHub(source, rssHubUrl, retryCount = 0) {
    const apiUrl = `https://api.rss2json.com/v1/api.json?rss_url=${encodeURIComponent(rssHubUrl)}`;
    const options = {
        muteHttpExceptions: true,
    };

    try {
        const response = UrlFetchApp.fetch(apiUrl, options);
        const data = response.getContentText();
        const jsonData = JSON.parse(data);

        if (!jsonData.items || !Array.isArray(jsonData.items)) {
            if (retryCount < 3) {
                return extractArticlesFromRSSHub(source, rssHubUrl, retryCount + 1);
            }
        }

        const newsData = [];
        const currentDate = new Date();
        const twentyFourHoursAgo = new Date(currentDate - 24 * 60 * 60 * 1000); // Calculate the date 24 hours ago

        for (const item of jsonData.items) {
            try {
                const title = item.title;
                let description = item.description;
                const link = item.link;

                if (description.length < 1000) {
                    const articleResponse = UrlFetchApp.fetch(link, options);
                    const articleHtml = articleResponse.getContentText();
                    description = extractFullContentFromHtml(articleHtml);
                }

                const keywords = getKeywords(title + " " + description)
                const embeddings = getOpenAIEmbedding(keywords);

                const pubDate = new Date(item.pubDate);
                if (pubDate >= twentyFourHoursAgo) {
                    newsData.push({
                        title: title,
                        description: description,
                        link: link,
                        embeddings: embeddings,
                    });
                }
            } catch (itemError) {
                continue;
            }
        }

        return newsData;
    } catch (error) {
        if (retryCount < 3) {
            return extractArticlesFromRSSHub(source, rssHubUrl, retryCount + 1);
        } else {
            console.error("Error: An exception occurred while fetching or processing RSS data.", error);
            return [];
        }
    }
}

function extractFullContentFromHtml(htmlContent) {
    // Remove script, style, and other non-article content
    let cleanedContent = htmlContent
        .replace(/<script[^>]*>[\s\S]*?<\/script>/gi, '')  // Remove <script> tags and their content
        .replace(/<style[^>]*>[\s\S]*?<\/style>/gi, '')  // Remove <style> tags and their content
        .replace(/<\/?[^>]+>/g, '')  // Remove all other HTML tags
        .replace(/&nbsp;/g, ' ')  // Replace non-breaking spaces with regular spaces
        .replace(/&amp;/g, '&')
        .replace(/&quot;/g, '"')
        .replace(/&apos;/g, "'")
        .replace(/&lt;/g, '<')
        .replace(/&gt;/g, '>')
        .replace(/\s+/g, ' ')  // Replace multiple spaces with a single space
        .trim();  // Remove leading and trailing spaces

    // Extract the article body, assuming it is within <article> tags or similar
    const articleMatch = cleanedContent.match(/<article[^>]*>([\s\S]*?)<\/article>/i);
    if (articleMatch && articleMatch[1]) {
        return articleMatch[1].trim();
    }

    return cleanedContent;
}

    
function extractHackerNews() {
    const now = new Date();
    const twentyFourHoursAgo = new Date(now - 24 * 60 * 60 * 1000);
    const rssUrl = 'https://hnrss.org/frontpage';
    const xmlResponse = UrlFetchApp.fetch(rssUrl);
    const xmlDoc = XmlService.parse(xmlResponse.getContentText());
    const items = xmlDoc.getRootElement().getChild("channel").getChildren("item");
    const newsData = [];

    for (let i = 0; i < items.length; i++) {
        const item = items[i];
        const pubDateStr = item.getChild("pubDate").getText();
        const pubDate = new Date(pubDateStr);

        if (pubDate >= twentyFourHoursAgo) {
            const title = item.getChild("title").getText();
            const link = item.getChild("link").getText();
            if (link === 'https://audio-video.gnu.org/video/gnu40/rms-gnu40.webm') {
                continue; // Skip this article and move to the next one
            }

            let contentHtml = "";
            try {
                const contentResponse = UrlFetchApp.fetch(link, { muteHttpExceptions: true });
                contentHtml = contentResponse.getContentText();
            } catch (contentError) {
                console.error("Error fetching content:", contentError);
                continue;
            }

            const description = extractDescriptionFromHtml(contentHtml);
            if (description.length < 400) {
                continue; // Skip this article if the description is too short
            }

            const keywords = getKeywords(title + " " + description)
            const embeddings = getOpenAIEmbedding(keywords);

            newsData.push({
                title: title,
                description: description,
                link: link,
                embeddings: embeddings,
            });
        }
    }
    return newsData;
}

function extractDescriptionFromHtml(contentHtml) {
    const metaDescriptionRegex = /<meta\s+name=["']description["']\s+content=["']([^"']+)["']\s*\/?>/i;
    const paragraphRegex = /<p[^>]*>(.*?)<\/p>/gi;
    const headingRegex = /<h[1-6][^>]*>(.*?)<\/h[1-6]>/gi;
    const listItemRegex = /<li[^>]*>(.*?)<\/li>/gi;
    const blockquoteRegex = /<blockquote[^>]*>(.*?)<\/blockquote>/gi;
    const divRegex = /<div[^>]*>(.*?)<\/div>/gi;
    const spanRegex = /<span[^>]*>(.*?)<\/span>/gi;
    const preRegex = /<pre[^>]*>(.*?)<\/pre>/gi;

    // Try to extract description from meta tag
    const metaMatch = contentHtml.match(metaDescriptionRegex);
    if (metaMatch) {
        return metaMatch[1].trim();
    }

    let description = '';

    // Helper function to clean and combine matched text
    function cleanAndCombine(matches) {
        return matches
            .map((match) => match.replace(/<[^>]+>/g, '').trim())
            .filter((text) => text.length > 0)
            .join(' ');
    }

    // Combine different text sources
    const allMatches = [
        ...(contentHtml.match(paragraphRegex) || []),
        ...(contentHtml.match(headingRegex) || []),
        ...(contentHtml.match(listItemRegex) || []),
        ...(contentHtml.match(blockquoteRegex) || []),
        ...(contentHtml.match(divRegex) || []),
        ...(contentHtml.match(spanRegex) || []),
        ...(contentHtml.match(preRegex) || [])
    ];

    description = cleanAndCombine(allMatches);

    const MIN_DESCRIPTION_LENGTH = 400;
    const MAX_DESCRIPTION_LENGTH = 1000;

    // Ensure the description is long enough by continuing to append content
    while (description.length < MIN_DESCRIPTION_LENGTH && allMatches.length > 0) {
        const nextBatch = allMatches.splice(0, 5); // Get next few items
        description += ' ' + cleanAndCombine(nextBatch);
    }

    // Trim the description to the maximum length if necessary
    if (description.length > MAX_DESCRIPTION_LENGTH) {
        description = description.substring(0, MAX_DESCRIPTION_LENGTH) + '...';
    }

    return description.trim();
}
    
function getArrayFromString(string){
  const array1 = string.split(",");
  const array = array1.map(parseFloat);

  return array;
}
    
function interestSearchArticles(interestEmbeddingsList, articles) {

  const interestArticles = [];
  const articlesList = [];
  for (const interestEmbedding of interestEmbeddingsList){
    const similarArticles = articles.map(article => {
      const articleEmbedding = article.embeddings;
      const similarity = calculateCosineSimilarity(interestEmbedding.embedding, articleEmbedding);
      const tag = interestEmbedding.tag;
      return { article,tag , similarity };
    });

    similarArticles.sort((a, b) => b.similarity - a.similarity);
    const topFiveSimilarArticles = similarArticles.slice(0, 5);
    articlesList.push(...topFiveSimilarArticles);

      /*generate the proper article for it{
          calculate the similarity between it and each article'embeddings
          return the most five similar articles each with its tag interest (tag: aritcle, ...)
          push them all in one list(articles)(each five corresponding to an interest)
      }*/
  }

  let counter1 = 0;
  let counter2 = 0;
  let listTitles = [];
  for (let x = 0; x < 5; x++) {
    if (counter2 >= articlesList.length - 1){
      return interestArticles;
    }
    if ( counter1 >= articlesList.length){
      counter2++;
      counter1 = counter2;
    }
    if (listTitles.includes(articlesList[counter1].article.title)){
      x--;
    } else {
      const object = { tag: articlesList[counter1].tag, article: articlesList[counter1].article };
      interestArticles.push(object);
      listTitles.push(articlesList[counter1].article.title);
    }
    counter1 += 5; // Increment by 5
  }  

  return interestArticles;
}
    
async function generateTitlesForUser(number, source, tags, news_text, link, let_count = 0, let_count2 = 0) {
  conversationPrompt = `Subject: Request for Generating Editorial News Title.
    Using the provided article's title, description,(${news_text}) generate captivating title, editorial title, interesting to the user (these are the user's interests: ${tags}) from its original title and its description. Do not use the same exact words, be creative and make the generated title attractive.

    Finally, return only the last version of the generated title without quoting it. formatting it as follows:
    - The newly final generated headline -.
    `;
  try {
    var api_response =await send_request(conversationPrompt);

    var matches1 = api_response.match(/- (.*?) -/);
    var matches2 = api_response.match(/-(.*?)-/);

    if (matches1 && matches1.length >= 1) {
      var extractedText = matches1[1];
      var generatedText = `${number}- ` + extractedText + `: <a href="${link}">${source}</a>`;
    } else if (matches2 && matches2.length >= 1) {
      var extractedText = matches2[1];
      var generatedText = `${number}- ` + extractedText + `: <a href="${link}">${source}</a>`;
    } else {
      var generatedText = `${number}- ` + api_response + `: <a href="${link}">${source}</a>`
    }

    if (!api_response) {
      if (let_count <= 6) {
        return generateTitlesForUser(number, source, tags, news_text, link, let_count + 1, let_count2);
      } else {
        console.error("Error: No titles and links found in API response or Empty API response.")
        return "Error: No titles and links found in API response or Empty API response.";
      }
    }
    return generatedText;
  } catch (error) {
    if (let_count2 <= 6) {
      return generateTitlesForUser(number, source, tags, news_text, link, let_count, let_count2 + 1);
    } else {
      console.error("Error: An exception occurred while generating titles.", error);
      return "No titles were found or generated.";
    }
  }
}
    
function send_request(prompt, let_count = 0) {
  var headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer ' + apiKey,
  };

  var requestBody = {
    model: 'gpt-3.5-turbo-1106',
    messages: [{ role: 'system', content: 'You are a helpfull assistant' }, { role: 'user', content: prompt }],
    max_tokens: 512,
    temperature: 0.70,
    top_p: 1.0,
    n: 1,
    stop: null,
  };

  var options = {
    method: 'post',
    headers: headers,
    payload: JSON.stringify(requestBody),
    muteHttpExceptions: true, // Capture the full response for examination
  };

  try {
    var response = UrlFetchApp.fetch(openAIUrl, options);

    // Check the status code of the response
    if (response.getResponseCode() !== 200) {
      if (let_count <= 3) {
        return send_request(prompt, let_count+1); // Retry the request
      } else {
        console.error("Error: Failed to fetch API response. Status code: " + response.getResponseCode());
        return "Error: Failed to fetch API response.";
      }
    }

    var data = JSON.parse(response.getContentText());

    if (!data || !data.choices || data.choices.length === 0 || !data.choices[0].message || !data.choices[0].message.content) {
      if (let_count <= 3) {
        return send_request(prompt, let_count+1); // Retry the request
      } else {
        console.error("Error: Invalid API response data format.");
        return "Error: Invalid API response data format.";
      }
    }

    var generated_text = data.choices[0].message.content;
    var html_response = "<html><body>" + generated_text + "</body></html>";
    return html_response;
  } catch (error) {
    console.error("Error: An exception occurred while fetching API response.", error);

    if (error.name === "Exception" && error.message.includes("Timeout")) {
      if (let_count <= 3) {
        return send_request(prompt, let_count+1); // Retry the request
      } else {
        return "Error: A timeout occurred while fetching API response.";
      }
    } else if (error.name === "Exception" && error.message.includes("URLFetch")) {
      if (let_count <= 3) {
        return send_request(prompt, let_count+1); // Retry the request
      } else {
        return "Error: Network error occurred during API request.";
      }
    } else if (let_count <= 3) {
      return send_request(prompt, let_count+1); // Retry the request
    } else {
      console.error("Error: An exception occurred while fetching API response.", error);
      return "Error: An exception occurred while fetching API response.";
    }
  }
}
    
async function sendemail(mailSubject, mailContent, recepientsMailList, user, welcome) {
      // Set the options for the email
      if(welcome == false){
        var body = `
          Dear ${user},<br>
          We hope you're having a great day!<br>
        `;
      } else if (welcome == true){
        var body = `
          Dear ${user},<br>
          Welcome to News Wave!<br>Get ready to receive personalized news titles based on your interests.<br><br>
        ` ;
      }
    
      body += mailContent;
      body += `<br><br>Stay informed and enjoy your day!If you have questions, please contact our support team.<br><br>
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
    
async function job() {
  const Fox_News = extractArticlesFromRSSHub('Fox News', 'https://feeds.foxnews.com/foxnews/latest');
  const The_New_York_Times = extractArticlesFromRSSHub('The New York Times', 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml');
  const The_Guardian = extractArticlesFromRSSHub('The Guardian', 'https://www.theguardian.com/world/rss');
  const Al_Jazeera = extractArticlesFromRSSHub('Al Jazeera', 'https://www.aljazeera.com/xml/rss/all.xml');
  const TechCrunch = extractArticlesFromRSSHub('TechCrunch', 'https://techcrunch.com/feed/');
  const BBC_News = extractArticlesFromRSSHub('BBC News', 'http://feeds.bbci.co.uk/news/rss.xml');
  const ESPN = extractArticlesFromRSSHub('ESPN', 'http://www.espn.com/espn/rss/news');
  const The_Verge = extractArticlesFromRSSHub('The Verge', 'https://www.theverge.com/rss/index.xml'); 
  const Time = extractArticlesFromRSSHub('Time', 'https://time.com/feed/');
  const Hacker_News = extractHackerNews(); 

  const feeds = [
    { source: 'Fox News', feed: Fox_News},
    { source: 'The New York Times', feed: The_New_York_Times},
    { source: 'The Verge', feed: The_Verge},
    { source: 'Hacker News', feed: Hacker_News},
    { source: 'The Guardian', feed: The_Guardian},
    { source: 'Al Jazeera', feed: Al_Jazeera},
    { source: 'TechCrunch', feed: TechCrunch},
    { source: 'BBC News', feed: BBC_News},
    { source: 'ESPN', feed: ESPN},
    { source: 'Time', feed: Time}
  ];

  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();

  const subjectDate = new Date().toISOString().slice(0, 10);
  const mailSubject = "Your Personalized Daily News Update - " + subjectDate;

  const columnNumber = 1;
  const columnData = sheet.getRange(1, columnNumber, sheet.getLastRow(), 1).getValues();
  const rowCount = columnData.filter(value => value[0] !== '').length;

  // Loop through each row of user data
  for (let i = 2; i < rowCount + 2; i++) {
    const range = sheet.getRange("C" + i);
    const valueEmail = range.getValue();
    const range2 = sheet.getRange("F" + i);
    const valueName = range2.getValue();
    const range3 = sheet.getRange("D" + i);
    const valueTags = range3.getValue().split(', ').map(source => source.trim());
    const range4 = sheet.getRange("E" + i);
    const valueSources = range4.getValue().split(',').map(source => source.trim());
    const range5 = sheet.getRange("G" + i); 
    const range6 = sheet.getRange("H" + i); 
    const range7 = sheet.getRange("I" + i); 
    const range8 = sheet.getRange("J" + i); 
    const range9 = sheet.getRange("K" + i); 
    var columnsList = [range5, range6, range7, range8, range9]

    const tagsAndEmbeddings = [];

    for (let j = 0; j < valueTags.length; j++) {
      const newObject = { tag: valueTags[j], embedding: getArrayFromString(columnsList[j].getValue()) };
      tagsAndEmbeddings.push(newObject);
    }

    const responsePromises = [];
    for (const articles of feeds) {    
      if (valueSources.includes(articles.source)) {
        responsePromises.push(`<br>Here are today's top stories tailored to your interests from ${articles.source}:`);
        var j = 1;
        var favoriteList = interestSearchArticles(tagsAndEmbeddings, articles.feed);
        for (const list of favoriteList) {
          var feed = 'title: ' + list.article.title + ' ' + 'description: ' + list.article.description;
          responsePromises.push(generateTitlesForUser(j, articles.source, list.tag, feed, list.article.link));
          j++;
        }
      }
    }
    const responses = await Promise.all(responsePromises);

    if (responses.length > 0) {
      sendemail(mailSubject, responses.join('<br>\n'), valueEmail, valueName, false); 
    }
  }
}
    
async function sendWelcomeEmail() {
  const Fox_News = extractArticlesFromRSSHub('Fox News', 'https://feeds.foxnews.com/foxnews/latest');
  const The_New_York_Times = extractArticlesFromRSSHub('The New York Times', 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml');
  const The_Guardian = extractArticlesFromRSSHub('The Guardian', 'https://www.theguardian.com/world/rss');
  const Al_Jazeera = extractArticlesFromRSSHub('Al Jazeera', 'https://www.aljazeera.com/xml/rss/all.xml');
  const TechCrunch = extractArticlesFromRSSHub('TechCrunch', 'https://techcrunch.com/feed/');
  const BBC_News = extractArticlesFromRSSHub('BBC News', 'http://feeds.bbci.co.uk/news/rss.xml');
  const ESPN = extractArticlesFromRSSHub('ESPN', 'http://www.espn.com/espn/rss/news');
  const The_Verge = extractArticlesFromRSSHub('The Verge', 'https://www.theverge.com/rss/index.xml');
  const Time = extractArticlesFromRSSHub('Time', 'https://time.com/feed/');
  const Hacker_News = extractHackerNews(); 

  const feeds = [
    { source: 'Fox News', feed: Fox_News},
    { source: 'The New York Times', feed: The_New_York_Times},
    { source: 'The Verge', feed: The_Verge},
    { source: 'Hacker News', feed: Hacker_News},
    { source: 'The Guardian', feed: The_Guardian},
    { source: 'Al Jazeera', feed: Al_Jazeera},
    { source: 'TechCrunch', feed: TechCrunch},
    { source: 'BBC News', feed: BBC_News},
    { source: 'ESPN', feed: ESPN},
    { source: 'Time', feed: Time}
  ];

  const sheet = SpreadsheetApp.getActiveSpreadsheet().getActiveSheet();
  const lastRow = sheet.getLastRow();
  const range = sheet.getRange("C" + lastRow);
  const valueEmail = range.getValue();
  const range2 = sheet.getRange("F" + lastRow); 
  const valueName = range2.getValue();
  const range3 = sheet.getRange("D" + lastRow); 
  const valueTags = range3.getValue().split(', ').map(source => source.trim());
  const range4 = sheet.getRange("E" + lastRow); 
  const valueSourcesRaw = range4.getValue();
  const valueSources = valueSourcesRaw.split(',').map(source => source.trim());
  const range5 = sheet.getRange("G" + lastRow); 
  const range6 = sheet.getRange("H" + lastRow); 
  const range7 = sheet.getRange("I" + lastRow); 
  const range8 = sheet.getRange("J" + lastRow); 
  const range9 = sheet.getRange("K" + lastRow); 

  const mailSubject = "Welcome to News Wave!";

  var columnsList = [range5, range6, range7, range8, range9]
  var embeddingTags = [];
  // trying the tags embeddings to insert them into the spreadsheet
  valueTags.forEach((item) => {
    let x = getOpenAIEmbedding(item).join(', ');
    embeddingTags.push(x);
  });

  for (let i = 0; i < columnsList.length; i++) {
    if (i >= embeddingTags.length){
      break
    }
    columnsList[i].setValue(embeddingTags[i]);
  }

  const tagsAndEmbeddings = [];

  for (let j = 0; j < valueTags.length; j++) {
    const newObject = { tag: valueTags[j], embedding: embeddingTags[j]};
    tagsAndEmbeddings.push(newObject);
  }

  const responsePromises = [];
  for (const articles of feeds) {    
    if (valueSources.includes(articles.source)) {
      responsePromises.push(`<br>Here are today's top stories tailored to your interests from ${articles.source}:`);
      var j = 1;
      var favoriteList = interestSearchArticles(tagsAndEmbeddings, articles.feed);
      for (const list of favoriteList) {
        var feed = 'title: ' + list.article.title + ' ' + 'description: ' + list.article.description;
        responsePromises.push(generateTitlesForUser(j, articles.source, list.tag, feed, list.article.link));
        j++;
      }
    }
  }
  const responses = await Promise.all(responsePromises);

  if (responses.length > 0) {
    sendemail(mailSubject, responses.join('<br>\n'), valueEmail, valueName, false); 
  }
}