library("stringr")
library("data.table")
library("dynlm")

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
         mutate(!!sym(paste0(x, "_lag", i)) := lag(!!sym(x), i)) %>% # nolint
          tidyr::drop_na()

        fit_lm <- lm(as.formula(paste0(y, "~ ", paste0(x, "_lag", i), " -1")), data = tmp_data) # nolint

        vnumber <- (as.integer(str_replace(x, "V", "")) + dim(data)[2] * ar_counter) # nolint
        tmp_out <- list(eq = as.character(str_replace(y, "V", "")),
                        variable = as.character(vnumber),
                        beta_estimate = fit_lm$coefficients[[1]])

        beta2x2[[counter]] <- tmp_out
        counter <- counter + 1
      }
    }
    ar_counter <- ar_counter + 1
  }

  beta2x2 <- do.call("rbind", beta2x2) %>% as.data.table()
  beta2x2 <- apply(beta2x2, 2, unlist)
  beta2x2 <- apply(beta2x2, 2, as.numeric) %>%
   as.data.table() %>%
    mutate(eq = as.character(eq), variable = as.character(variable))

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
         mutate(!!sym(paste0(x, "_lag", i)) := lag(!!sym(x), i)) %>% # nolint
          tidyr::drop_na()

        estimate_corr <- cor(tmp_data %>% dplyr::select(!!sym(y), !!sym(paste0(x, "_lag", i))) %>% tidyr::drop_na()) # nolint
        estimate_corr <- estimate_corr[1, 2]

        vnumber <- (as.integer(str_replace(x, "V", "")) + dim(data)[2] * ar_counter) # nolint
        tmp_out <- list(eq = as.character(str_replace(y, "V", "")),
                        variable = as.character(vnumber),
                        corr_estimate = estimate_corr)

        corr[[counter]] <- tmp_out
        counter <- counter + 1
      }
    }
    ar_counter <- ar_counter + 1
  }

  corr <- do.call("rbind", corr) %>% as.data.table()
  corr <- apply(corr, 2, unlist)
  corr <- apply(corr, 2, as.numeric) %>%
   as.data.table() %>%
    mutate(eq = as.character(eq), variable = as.character(variable))

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
         mutate(!!sym(paste0(x, "_lag", i)) := lag(!!sym(x), i)) %>% # nolint
          tidyr::drop_na()

        estimate_cov <- cov(tmp_data %>% dplyr::select(!!sym(y), !!sym(paste0(x, "_lag", i))) %>% tidyr::drop_na()) # nolint
        estimate_cov <- estimate_cov[1, 2]

        vnumber <- (as.integer(str_replace(x, "V", "")) + dim(data)[2] * ar_counter) # nolint
        tmp_out <- list(eq = as.character(str_replace(y, "V", "")),
                        variable = as.character(vnumber),
                        cov_estimate = estimate_cov)

        cov[[counter]] <- tmp_out
        counter <- counter + 1
      }
    }
    ar_counter <- ar_counter + 1
  }

  cov <- do.call("rbind", cov) %>% as.data.table()
  cov <- apply(cov, 2, unlist)
  cov <- apply(cov, 2, as.numeric) %>%
   as.data.table() %>%
    mutate(eq = as.character(eq), variable = as.character(variable))

  return(cov)
}