library("stringr")
library("data.table")

longitudinal_to_data.table = function(data){
  
  data_out = list()
  for (i in 1:dim(data)[1]){
    data_out[[i]] = data[,i]
  }
  data_out = do.call("cbind", data_out) %>% as.data.table()
  
  return(data_out)
}

lm_combination_2x2 = function(data){
  
  beta2x2 = list()
  counter = 1
  for (y in colnames(data)){
    for (x in colnames(data)){
      if (y == x){
        tmp_data = list(Var1=str_replace(y, "V", ""), 
                        Var2=str_replace(x, "V", ""),
                        beta_2x2=1)
      }
      else{
        fit_lm = lm(as.formula(paste0(y, "~" , x, "-1")), data = data)
        tmp_data = list(Var1=str_replace(y, "V", ""),
                        Var2=str_replace(y, "V", ""),
                        beta_2x2=fit_lm$coefficients[[1]])
      }
      
      beta2x2[[counter]] = tmp_data
      counter = counter + 1
    }
  }
  beta2x2 = do.call("rbind", beta2x2) %>% as.data.table()
  beta2x2 = apply(beta2x2, 2, unlist)
  beta2x2 = apply(beta2x2, 2, as.numeric) %>% as.data.table()
  
  return(beta2x2)
}