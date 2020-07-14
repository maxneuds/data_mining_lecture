library(keras)
# 
#load(file = "data/clean_data.RData")

clean_data = function(data) {
    data$X = NULL
    data$Date = NULL
    data$Evaporation = NULL
    data$Sunshine = NULL
    data$WindDir3pm = NULL
    data$WindDir9am = NULL
    data$WindGustDir = NULL
    data$RainTomorrow = NULL
    data$RainToday = ifelse(data$RainToday == "Yes", 1, 0)
    
    data = na.omit(data)
    
    return(data)
}

darwin_train = read.csv('data/darwin_train.csv')

x_train = clean_data(darwin_train)

x_test = read.csv('data/darwin_test.csv')

x_test = clean_data(darwin_test)

# https://blogs.rstudio.com/tensorflow/posts/2017-12-20-time-series-forecasting-with-recurrent-neural-networks/

# parms
# loopback = 7 Tage zurück
# step = 1 jeden Tag neue Daten ziehen
# delay = 1 für einen Tag prognose machen
# batch_size = 1 was auch immer das genau soll
lookback <- 5
step <- 1
delay <- 1
batch_size = 128

# combine data
data_train = data.matrix(x_train)
data_test = data.matrix(x_test)

# normalize data
# train_data <- data_train[1:2000,]
# mean <- apply(train_data, 2, mean)
# std <- apply(train_data, 2, sd)
# data_train <- scale(data_train, center = mean, scale = std)
# data_test <- scale(data_test, center = mean, scale = std)

# create generator
generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 1) {
    if (is.null(max_index))
        max_index <- nrow(data) - delay - 1
    i <- min_index + lookback
    function() {
        if (shuffle) {
            rows <- sample(c((min_index+lookback):max_index), size = batch_size)
        } else {
            if (i + batch_size >= max_index)
                i <<- min_index + lookback
            rows <- c(i:min(i+batch_size-1, max_index))
            i <<- i + length(rows)
        }
        
        samples <- array(0, dim = c(length(rows),
                                    lookback / step,
                                    dim(data)[[-1]]))
        targets <- array(0, dim = c(length(rows)))
        
        for (j in 1:length(rows)) {
            indices <- seq(rows[[j]] - lookback, rows[[j]]-1,
                           length.out = dim(samples)[[2]])
            samples[j,,] <- data[indices,]
            targets[[j]] <- data[rows[[j]] + delay, ncol(data)]
        }           
        list(samples, targets)
    }
}

train_gen <- generator(
    data_train,
    lookback = lookback,
    delay = delay,
    min_index = 1,
    max_index = 2000,
    step = step, 
    batch_size = batch_size
)

val_gen = generator(
    data_train,
    lookback = lookback,
    delay = delay,
    min_index = 2001,
    max_index = NULL,
    step = step,
    batch_size = batch_size
)

test_gen <- generator(
    data_test,
    lookback = lookback,
    delay = delay,
    min_index = 1,
    max_index = NULL,
    step = step,
    batch_size = batch_size
)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps <- (nrow(data_train) - 2001 - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps <- (nrow(data_test) - 1 - lookback) / batch_size

# clear session data
k_clear_session()

# create model
shape = input_shape = c(lookback / step, dim(data)[-1])
shape = input_shape = list(NULL, dim(data_train)[[-1]])
model = keras_model_sequential()
# layer_dense(model, units = 2, input_shape = shape)
layer_lstm(model, units = 50, input_shape = shape, batch_size = batch_size, stateful = TRUE)
layer_dropout(model, rate = 0.2)
layer_dense(model, units = 1, activation = 'sigmoid')

compile(model,
        loss = 'binary_crossentropy',
        optimizer = 'adam',
        metrics = c('accuracy')
)

summary(model)

# train model
history = fit_generator(
    model,
    train_gen,
    steps_per_epoch = 2000 / batch_size,
    epochs = 30,
    validation_data = val_gen,
    validation_steps = val_steps
)

tg = test_gen()[1]
out = predict(model, tg, batch_size = batch_size)

# save model
save_model_hdf5(model, "lstm_darwin_rain.h5")











