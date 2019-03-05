#!/usr/bin/env Rscript

library('tm')
library('ggplot2')

# Set global paths
spam_training <- file.path("data", "myspam")
ham_training  <- file.path("data", "myham")
testing_data  <- file.path("data", "mytest")

# Return a single element vector of just the email body
# This is a very simple approach, as we are only using
# words as features
get_msg <- function(path)
{
  con <- file(path, open = "rt", encoding = "latin1")
  text <- readLines(con)
  msg <- text[seq(1, length(text), 1)]
  close(con)
  return(paste(msg, collapse = "\n"))
}

# Create a TermDocumentMatrix (TDM) from the corpus of SPAM email.
# The TDM control can be modified, and the sparsity level can be
# altered.  This TDM is used to create the feature set used to do
# train our classifier.
get_tdm <- function(doc.vec)
{
  control <- list(stopwords = TRUE,
                  removePunctuation = TRUE,
                  removeNumbers = TRUE,
                  minDocFreq = 2)
  doc.corpus <- Corpus(VectorSource(doc.vec))
  doc.dtm <- TermDocumentMatrix(doc.corpus, control)
  return(doc.dtm)
}

# This is the our workhorse function for classifying email.  It takes
# two required parameters: a file path to an email to classify, and
# a data frame of the trained data.  The function also takes two
# optional parameters.  First, a prior over the probability that an email
# is SPAM, which we set to 0.5 (naive), and constant value for the
# probability on words in the email that are not in our training data.
# The function returns the naive Bayes probability that the given email
# is SPAM.
classify_email <- function(path, training.df, prior = 0.5, c = 1e-2)
{
  # Here, we use many of the support functions to get the
  # email text data in a workable format
  msg <- get_msg(path)
  msg.tdm <- get_tdm(msg)
  msg.freq <- rowSums(as.matrix(msg.tdm))
  # Find intersections of words
  msg.match <- intersect(names(msg.freq), training.df$term)
  # Now, we just perform the naive Bayes calculation
  if(length(msg.match) < 1)
  {
    return(prior * c ^ (length(msg.freq)))
  }
  else
  {
    match.probs <- training.df$occurrence[match(msg.match, training.df$term)]
    return(prior * prod(match.probs) * c ^ (length(msg.freq) - length(msg.match)))
  }
}

###############################################################################
# With all of our support functions written, we can perform the classification.
###############################################################################

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Build training data for SPAM emails
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

spam.docs <- dir(spam_training)
all.spam <- sapply(spam.docs, function(p) get_msg(file.path(spam_training, p)))

# Create a DocumentTermMatrix from that vector
spam.tdm <- get_tdm(all.spam)

# Create a data frame that provides the feature set from the training SPAM data
spam.matrix <- as.matrix(spam.tdm)
spam.counts <- rowSums(spam.matrix)

spam.df <- data.frame(cbind(names(spam.counts),
                      as.numeric(spam.counts)),
                      stringsAsFactors = FALSE)
names(spam.df) <- c("term", "frequency")
spam.df$frequency <- as.numeric(spam.df$frequency)

spam.occurrence <- sapply(1:nrow(spam.matrix),
                          function(i)
                          {
                            length(which(spam.matrix[i, ] > 0)) / ncol(spam.matrix)
                          })

# Add term occurrence rate
spam.df <- transform(spam.df, occurrence = spam.occurrence)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Build training data for HAM emails
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

ham.docs <- dir(ham_training)
all.ham <- sapply(ham.docs[1:length(ham.docs)],
                      function(p) get_msg(file.path(ham_training, p)))

ham.tdm <- get_tdm(all.ham)
ham.matrix <- as.matrix(ham.tdm)

ham.counts <- rowSums(ham.matrix)
ham.df <- data.frame(cbind(names(ham.counts),
                         as.numeric(ham.counts)),
                         stringsAsFactors = FALSE)
names(ham.df) <- c("term", "frequency")
ham.df$frequency <- as.numeric(ham.df$frequency)
ham.occurrence <- sapply(1:nrow(ham.matrix),
                          function(i)
                          {
                            length(which(ham.matrix[i, ] > 0)) / ncol(ham.matrix)
                          })

# Add term occurrence rate
ham.df <- transform(ham.df, occurrence = ham.occurrence)

# Show data frames for SPAM and HAM

# head( ham.df[with(ham.df,order(-frequency)),],  n=10 )
# head( ham.df[with(ham.df,order(-occurrence)),], n=10 )

# head( spam.df[with(spam.df,order(-frequency)),],  n=10 )
# head( spam.df[with(spam.df,order(-occurrence)),], n=10 )

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# Run classifier
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 1. Load testing email(s)
test.docs <- dir(testing_data)
test.docs = test.docs[1]

# 2. Compute probabilities against spam dataset
spam.test <- sapply(test.docs,
                          function(p) classify_email(file.path(testing_data, p),
                                      training.df = spam.df))

# 3. Compute probabilities against ham dataset
ham.test <- sapply(test.docs,
                          function(p) classify_email(file.path(testing_data, p),
                                      training.df = ham.df))

if (spam.test<ham.test){
  print('Test was HAM :)')
} else {
  print('Test was SPAM :(')
}
