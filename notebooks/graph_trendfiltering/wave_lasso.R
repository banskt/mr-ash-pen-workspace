rm(list=ls())
library(genlasso
        )

set.seed(1)
 n = 100
i = 1:n
y = (i > 20 & i < 30) + 5*(i > 50 & i < 70)   + rnorm(n, sd=0.1)

out = fusedlasso1d(y)
out
plot(out)

set.seed(0)
edges = c()
tt1 <-0
#### standard connection parent children  -----
for ( j in 0:5)
{
  for ( k in 0:((2^j)-1)){
    tt <- c( k+(2^j),(2*k)+2^(j+1) ,(k)+2^j,(2*k+1)+2^(j+1) )


    if(!(mean(tt1==tt)==1))
    {

      edges <- c(edges,tt)

    }
    tt1 <-tt
    }


}
#edges <- edges -1


gr = graph(edges=edges,directed=FALSE)
D = getDgSparse(gr)
D
plot(gr)

library(wavethresh)
test_func <- wavethresh::DJ.EX(n = 128, rsnr = 1, noisy = TRUE )
f0        <- test_func[[1]]

w0 <- wavethresh::wd(f0)
w0$D
sW0 <- sign(w0$D)
max(edges)

library(wavethresh)
a1 = fusedlasso(   w0$D ,graph=gr, gamma=1)

a2 = fusedlasso(   sW0*w0$D ,graph=gr, gamma=1)



plot(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]], type="l")
w0$D <-  a1$fit[,20]
lines(wr(w0), col="blue")
w0$D <- sW0*a2$fit[,20]
lines(wr(w0), col="red")
lines(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]])





edges = c()
tt1 <-0
#### connection parent children and between siblings -----
for ( j in 0:5)
{
  for ( k in 0:((2^j)-1)){


      tt <- c( k+(2^j),(2*k)+2^(j+1) ,(k)+2^j,(2*k+1)+2^(j+1),
               (2*k)+2^(j+1),(2*k+1)+2^(j+1))
      #connection parent children
      #children children



    if(!(mean(tt1==tt)==1))
    {

      edges <- c(edges,tt)

    }
    tt1 <-tt
  }


}
#edges <- edges -1


gr = graph(edges=edges,directed=FALSE)
D = getDgSparse(gr)
D
plot(gr)

library(wavethresh)
test_func <- wavethresh::DJ.EX(n = 128, rsnr = 1, noisy = TRUE )
f0        <- test_func[[1]]

w0 <- wavethresh::wd(f0)
w0$D
max(edges)

library(wavethresh)
a1 = fusedlasso(   w0$D ,graph=gr, gamma=1)

a2 = fusedlasso(   y=sW0*w0$D ,graph=gr , gamma=1)



plot(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]], type="l")
w0$D <-  a1$fit[,50]
lines(wr(w0), col="blue")
w0$D <- sW0*a2$fit[,50]
lines(wr(w0), col="red")
lines(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]])
lines(smash(f0), col="green")




edges = c()
tt1 <-0
#### connection parent children and between siblings and between adjacent cousins-----
for ( j in 0:5)
{
  for ( k in 0:((2^j)-1)){

    #connection parent children
    #children children
    #parent cousin
    if(k <((2^j)-1))
    {
      tt <- c( k+(2^j),(2*k)+2^(j+1) ,(k)+2^j,(2*k+1)+2^(j+1),#paretn children
               k+(2^j),(2*(k+1))+2^(j+1) ,(k)+2^j,(2*(k+1)+1)+2^(j+1),#parent couson
               (2*k)+2^(j+1),(2*k+1)+2^(j+1)#children children

      )


    }else{
      tt <- c( k+(2^j),(2*k)+2^(j+1) ,(k)+2^j,(2*k+1)+2^(j+1),
               (2*k)+2^(j+1),(2*k+1)+2^(j+1))

    }



    if(!(mean(tt1==tt)==1))
    {

      edges <- c(edges,tt)

    }
    tt1 <-tt
  }


}
#edges <- edges -1


gr = graph(edges=edges,directed=FALSE)
D = getDgSparse(gr)
D
plot(gr)

library(wavethresh)
test_func <- wavethresh::DJ.EX(n = 128, rsnr = 1, noisy = TRUE )
f0        <- test_func[[1]]

w0 <- wavethresh::wd(f0)
w0$D
max(edges)

library(wavethresh)
a1 = fusedlasso(   w0$D ,graph=gr, gamma=1)

a2 = fusedlasso(   sW0*w0$D ,graph=gr, gamma=1)



plot(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]], type="l")
w0$D <-  a1$fit[,50]
lines(wr(w0), col="blue")
w0$D <- sW0*a2$fit[,50]
lines(wr(w0), col="red")
lines(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]])
lines(smash(f0), col="green")




edges = c()
tt1 <-0
####  connection parent children and between siblings and between  cousins- -----
for ( j in 0:5)
{
  for ( k in 0:((2^j)-1)){

    #connection parent children
    #children children
    #parent cousin
    #left cousin right cousin
    if(k <((2^j)-1))
    {
      tt <- c( k+(2^j),(2*k)+2^(j+1) ,(k)+2^j,(2*k+1)+2^(j+1),#paretn children
               k+(2^j),(2*(k+1))+2^(j+1) ,(k)+2^j,(2*(k+1)+1)+2^(j+1),#parent couson
               (2*k)+2^(j+1),(2*k+1)+2^(j+1),#children children
               (2*k+1)+2^(j+1),(2*(k+1))+2^(j+1)
               )

    }else{
      tt <- c( k+(2^j),(2*k)+2^(j+1) ,(k)+2^j,(2*k+1)+2^(j+1),
               (2*k)+2^(j+1),(2*k+1)+2^(j+1))

    }



    if(!(mean(tt1==tt)==1))
    {

      edges <- c(edges,tt)

    }
    tt1 <-tt
  }


}
#edges <- edges -1


gr = graph(edges=edges,directed=FALSE)
D = getDgSparse(gr)
D
plot(gr)

library(wavethresh)
test_func <- wavethresh::DJ.EX(n = 128, rsnr = 1, noisy = TRUE )
f0        <- test_func[[1]]

w0 <- wavethresh::wd(f0)
w0$D
max(edges)

library(wavethresh)
a1 = fusedlasso(   w0$D ,graph=gr, gamma=1)

a2 = fusedlasso(   sW0*w0$D ,graph=gr, gamma=1)



plot(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]], type="l")
w0$D <-  a1$fit[,100]
lines(wr(w0), col="blue")
w0$D <- sW0*a2$fit[,200]
lines(wr(w0), col="red")
lines(wavethresh::DJ.EX(n = 128,   noisy = FALSE )[[1]])
lines(smash(f0), col="green")

