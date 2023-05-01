library("stringr")
library("data.table")
library("dynlm")

stardize_var_names <- function(k){
  new_names <- c()
  ks <- k

  for (k in 1:ks){
    new_names <- append(new_names, paste0("V", k))
  }
  
  return(new_names)
}

stardize_var_lag_names <- function(k, p){
  new_names <- c()
  ks <- k
  ps <- p
  
  for (k in 1:ks){
    for (p in 1:ps){
      new_names <- append(new_names, paste0("V", k, ".L", p))
    }
  }
  
  return(new_names)
}

longitudinal_to_data.table <- function(data) {

  data_out <- list()
  for (i in seq_len(dim(data)[2])) {
    data_out[[i]] <- data[, i]
  }
  data_out <- do.call("cbind", data_out) %>% as.data.table()

  return(data_out)
}

lm_combination_2x2 <- function(data, p) {

  beta2x2 <- list()
  ar_counter <- 0
  counter <- 1

  for (i in seq_len(p)){
    for (y in colnames(data)) {
      for (x in colnames(data)) {
        tmp_data <- data %>%
         mutate(!!sym(paste0(x, ".L", i)) := lag(!!sym(x), i)) %>%
          tidyr::drop_na()

        fit_lm <- lm(as.formula(paste0(y, "~ ", paste0(x, ".L", i), " -1")), data = tmp_data)

        tmp_out <- list(eq = y,
                        variable = paste0(x, ".L", i),
                        beta_estimate = fit_lm$coefficients[[1]])

        beta2x2[[counter]] <- tmp_out
        counter <- counter + 1
      }
    }
    ar_counter <- ar_counter + 1
  }

  beta2x2 <- do.call("rbind", beta2x2) %>% apply(2, unlist) %>%
    as.data.table() %>% mutate(eq = as.character(eq), variable = as.character(variable), beta_estimate=as.numeric(beta_estimate))

  return(beta2x2)
}

corr_combination_2x2 <- function(data, p) {

  corr <- list()
  ar_counter <- 0
  counter <- 1

  for (i in seq_len(p)){
    for (y in colnames(data)) {
      for (x in colnames(data)) {
        tmp_data <- data %>%
         mutate(!!sym(paste0(x, ".L", i)) := lag(!!sym(x), i)) %>% # nolint
          tidyr::drop_na()

        estimate_corr <- cor(tmp_data %>% dplyr::select(!!sym(y), !!sym(paste0(x, ".L", i))) %>% tidyr::drop_na()) # nolint

        tmp_out <- list(eq = colnames(estimate_corr)[1],
                        variable = colnames(estimate_corr)[2],
                        corr_estimate = estimate_corr[1, 2])

        corr[[counter]] <- tmp_out
        counter <- counter + 1
      }
    }
    ar_counter <- ar_counter + 1
  }
  
  corr <- do.call("rbind", corr) %>% apply(2, unlist) %>%
    as.data.table() %>% mutate(eq=as.character(eq), variable=as.character(variable), corr_estimate=as.numeric(corr_estimate))

  return(corr)
}

cov_combination_2x2 <- function(data, p) {

  cov <- list()
  ar_counter <- 0
  counter <- 1

  for (i in seq_len(p)){
    for (y in colnames(data)) {
      for (x in colnames(data)) {
        
        tmp_data <- data %>%
         mutate(!!sym(paste0(x, ".L", i)) := lag(!!sym(x), i)) %>%
          tidyr::drop_na()

        estimate_cov <- cov(tmp_data %>% dplyr::select(!!sym(y), !!sym(paste0(x, ".L", i))) %>% tidyr::drop_na()) # nolint

        tmp_out <- list(eq = colnames(estimate_cov)[1],
                        variable = colnames(estimate_cov)[2],
                        cov_estimate = estimate_cov[1, 2])

        cov[[counter]] <- tmp_out
        counter <- counter + 1
      }
    }
  }

  cov <- do.call("rbind", cov) %>% apply(2, unlist) %>%
    as.data.table() %>% mutate(eq=as.character(eq), variable=as.character(variable), cov_estimate=as.numeric(cov_estimate))

  return(cov)
}