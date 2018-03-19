----------------------
-- 2018.1
-- sule
-- mnist neutral network for AI course homework
----------------------
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE TupleSections #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE FlexibleContexts #-}

module Main where

import Data.List(foldl')  
import qualified Data.Array as A
import Control.Monad
import qualified Data.ByteString.Char8 as B
import qualified Data.ByteString.Lazy.Char8 as L
import Data.Serialize
import qualified Data.Vector.Storable as V
import Grenade
import Numeric.LinearAlgebra (maxIndex)
import qualified Numeric.LinearAlgebra.Static as S
import System.Random
import System.Directory
import Graphics.Rendering.Chart.Easy
import Graphics.Rendering.Chart.Backend.Diagrams(toFile)
import qualified Data.Array.IO as MA


-- |type of input and output of the network
type IOShape = (S ('D2 28 28), S ('D1 10))

-- |type of the neutral network
type HandRC
  = Network
  '[ Convolution 1 12 5 5 1 1, Pooling 2 2 2 2, Relu
     , Convolution 12 24 5 5 1 1, Pooling 2 2 2 2
     , Reshape, Relu
     , FullyConnected 384 80, Logit
     , FullyConnected 80 10, Logit]
  '[ 'D2 28 28, 'D3 24 24 12, 'D3 12 12 12, 'D3 12 12 12
     , 'D3 8 8 24, 'D3 4 4 24, 'D1 384, 'D1 384
     , 'D1 80, 'D1 80, 'D1 10, 'D1 10]

-- |parse the input and label from a csv file     
parseData :: FilePath -> IO [IOShape]
parseData path = do
  strs <- tail . L.lines <$!> L.readFile path
  return (map proc strs)
  where
    proc l =
      ( fromJust $! fromStorable $! V.fromList $! tail lbs
      , fromJust $! fromStorable $! V.fromList $! conv $! head lbs)
      where
        lbs = map (read . L.unpack) $! L.split ',' l
    fromJust (Just a) = a
    conv x' =
      let x = floor x'
      in replicate x 0 ++ [1] ++ replicate (9 - x) 0     

-- |train the network with a pair of input and label      
trainOne :: LearningParameters -> HandRC -> IOShape -> HandRC
trainOne rate !net p@(d, _) = uncurry (train rate net) p

-- |run the network, get the predict
run :: HandRC -> S ('D2 28 28) -> S ('D1 10)
run = runNet

-- |helper function for random sort
randomSortT :: [Int] -> IO [Int]
randomSortT [] = return []
randomSortT !ls = do
  a <- randomRIO (0, length ls - 1)
  (\ls' -> (ls !! a) : ls') <$!> (randomSortT $! (take a ls ++ drop (a + 1) ls))

-- |helper function for random sort  
concatBy::[Int]->[[a]]->[a]
concatBy [] _ = []
concatBy (x:xs) as = (as!!x)++concatBy xs as

-- |implement of random sort
randomSort :: [Int]->Int -> IO [Int]
randomSort ls step=
  if  lg<= step
    then randomSortT ls
    else do
      let ts = lg `div` step
      xr<-randomSortT [0..ts-1]
      res''<-mapM (randomSortT . get) [1 .. div (length ls) step]
      let res'=concatBy xr res''
      if lg `mod` step==0
        then return res'
        else do
          t<-randomRIO (0,ts)
          res<-randomSort (drop (ts*step) ls) step
          return $! take t res'++ res ++drop t res'
  where
    get x = drop (step * (x - 1)) $! take (step * x) ls
    lg =length ls

-- |train one generation    
trainOnce ::
     Int->LearningParameters -> Int -> [IOShape] -> [Int] -> HandRC -> IO HandRC
trainOnce gx lp time dts ls !net = do
  let lgT = length ls
  print $! lgT
  foldM
    (\n i -> do
        let ir = ls !! i
        print $! "training the " ++ show i ++ " th data " ++ show ir
        when (i `mod` gx==0) $! do
          let trained = take i dts
          let testD = drop lgT dts
          print $! "testing " ++ show i ++ " records trained before"
          let trainCR = validateList n trained
          appendFile "trained" $! Prelude.show (i, correctRate trainCR)++"\n"
          print $! "testing " ++ show (length dts-lgT) ++ " records for validate"
          let trainTest = validateList n testD
          appendFile "test" $! Prelude.show (i, correctRate trainTest)++"\n"
          plotFromFile ("trained", "test")
        return $!
         trainOne (lp {learningRate = learningRate lp * 0.7 ^ time}) n (dts !! ir))
    net
    [0 .. lgT - 1]

-- |train the nerwork with the given init rate, generations, time to calculate accuracy, csv path, and proportion to train
-- example: trainT 0.007 15 1000 "train.csv" 0.8 
-- means train the "train.csv" with a init rate of 0.007, keep training for 15 generations, per 1000 records calculate accuracy 
-- if claim a pre-trained network in the directory, it will be loaded
trainT :: FilePath->Double->Int ->Int-> FilePath -> Double -> IO ()
trainT preNet initRate time gx path trainS = do
  dts <- parseData path
  let lg = length dts - 1
  let lgT = 1 + floor (trainS * fromIntegral lg)
  bl<-doesFileExist preNet
  net <- if bl 
          then print"loaded">>load preNet
          else randomNetwork 
  let lp = LearningParameters initRate 0.9 0.0005
  net' <-
    foldM
      (\nt t -> do
         lsT <- randomSort [0 .. lgT-1] 5
         lsV <- randomSort [lgT .. lg] 5
         trainOnce gx lp t dts lsT nt)
      net
      [1 .. time]
  save net' "net"

calcConfuseMatrix::IO ()
calcConfuseMatrix = do
  net<-load "net"
  dts<-parseData "train.csv"
  ar<-MA.freeze=<<confuseMatrix net dts::IO (A.Array (Int,Int) Int)
  writeFile "confuseMatrix" $ renderA ar

renderA::A.Array (Int,Int) Int->String
renderA = foldl' (\str ((_,a),e)->if a==0 then str++"\n"++show e else str++" "++show e) [] . A.assocs 

-- |calculate the confuseMatrix using the given network and records
confuseMatrix::HandRC->[IOShape]->IO (MA.IOUArray (Int,Int) Int)
confuseMatrix net ls = do
  ar<-MA.newArray ((0,0),(9,9)) 0
  mapM_ (\((dt,lb),i)->do
    print i
    let i=(getIdx lb, getIdx $run net dt)
    x<-MA.readArray ar i
    MA.writeArray ar i $x+1) $zip ls [1..]
  return ar

-- |k-cross validation  
kCross::Int->Int->Double->FilePath->IO ()
kCross gen k initRate path = do
  lds<-parseData path
  let ws = length lds `div` k
  mapM_ (cross lds ws) [0..k-1]
  where cross lds ws t= do
        let w=t*ws
        let trainD = take w lds++drop (w+ws) lds
        let valD = drop w $! take (w+ws) lds
        net<-trainList initRate t gen trainD
        let tr=validateList net trainD
        let vr=validateList net valD
        save net ("k-cross-net-"++show t)
        putStrLn $!"Train "++show (correctRate tr)++" Validation "++show (correctRate vr)
        writeFile ("k-cross-accuracy-"++show t) $!"Train "++show (correctRate tr)++" Validation "++show (correctRate vr)

-- |helper function for k-cross        
justTrain :: LearningParameters -> Int -> [IOShape] -> HandRC -> IO HandRC
justTrain lp time dts !net = do
  let lgT = length dts 
  foldM
    (\n i -> do
        print $! "training the " ++ show i ++ " th data"
        return $!
         trainOne (lp {learningRate = learningRate lp * 0.9 ^ time}) n (dts !! i))
    net
    [0 .. lgT - 1]

-- |train a list of records, helper function for k-cross       
trainList :: Double->Int->Int->[IOShape] -> IO HandRC
trainList initRate r gen dts = do
  let lg = length dts
  print lg
  net <- randomNetwork :: IO HandRC
  let lp = LearningParameters initRate 0.9 0.0005
  foldM
      (\nt t -> do
         print $! "generation "++show t
         justTrain lp t dts nt)
      net
      [1 .. gen]

-- |get the correctRate from a list of bool      
correctRate :: [Bool] -> Double
correctRate [] = 0
correctRate !bs =
  fromIntegral (length (filter id bs)) / fromIntegral (length bs)

-- |validate a list of records  
validateList :: HandRC -> [IOShape] -> [Bool]
validateList _ [] = []
validateList !net !ls = map (validateOne net) ls

-- |validate one records  
validateOne :: HandRC -> IOShape -> Bool
validateOne !net (!dt, !lb) = judge (run net dt) lb

validate :: IOShape -> IOShape -> Bool
validate (_, !lb') (_, !lb) = judge lb' lb

judge :: S ('D1 10) -> S ('D1 10) -> Bool
judge !x !y = getIdx x == getIdx y

getIdx :: S ('D1 10) -> Int
getIdx (S1D !a) = maxIndex (S.extract a)

-- |save a network to file
save :: HandRC -> FilePath -> IO ()
save net path = B.writeFile path $! runPut (put net)

load :: FilePath -> IO HandRC
load path = do
  modelData <- B.readFile path
  either fail return $! runGet (get :: Get HandRC) modelData

plotFromFile::(FilePath,FilePath)->IO ()
plotFromFile (t,v)=do
  ts<-map read.lines<$>readFile t::IO [(Int,Double)]
  vs<-map read.lines<$>readFile t::IO [(Int,Double)]
  plotAccuracy (trans ts) "train accuracy" "train-accuracy.svg"
  plotAccuracy (trans vs) "validate accuracy" "validate-accuracy.svg"
  where trans xs = filter (\(a,b)->b/=0) $map (\((a,b),z)->(a+z,b)) $ zip xs $concat [replicate 34 x| x<-[0,34000..]]

plotAccuracy::[(Int,Double)]->String->FilePath->IO ()
plotAccuracy xs am path = toFile def path $ do
    layout_title .= "Accuracy"
    setColors [opaque green]
    plot (line am [xs])


runT::IO ()
runT = do
  putStrLn "Enter the pre-trained neutral network file(if not exist then init randomly)"
  net<-getLine
  putStrLn "Enter the init learning rate"
  initRate<-read<$>getLine
  putStrLn "Enter the generations to train(how many repetition)"
  time<-read<$>getLine
  putStrLn "Enter the duration of validation"
  gx<-read<$>getLine
  putStrLn "Enter the percentage for training(1-p for validation)"
  trainS<-read<$>getLine
  putStrLn "Enter the path of training data(train.csv)"
  path<-getLine
  trainT net initRate time gx path trainS 


runK::IO ()
runK = do
  putStrLn "Enter the k for k-cross validation"
  k<-read<$>getLine
  putStrLn "Enter the init learning rate"
  initRate<-read<$>getLine
  putStrLn "Enter the generations to train(how many repetition)"
  time<-read<$>getLine
  putStrLn "Enter the path of training data(train.csv)"
  path<-getLine
  kCross time k initRate path

main = do
  putStrLn "choose one action"
  putStrLn "0 -- run randomly training and validation"
  putStrLn "1 -- run k cross validation"
  x<-read<$>getLine::IO Int
  if x==0 
    then runT
    else if x==1 
      then runK
      else putStrLn "wrong option">>main
