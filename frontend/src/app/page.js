import { StockChart } from "@/components/StockChart";
import { PredictionPrice } from "@/components/PredictionPrice";
import { NewsArticle, NewsArticleList } from "@/components/NewsArticle";

export default function Home() {
  //const [stock, setStock] = React.useState("ALIBABA");

  return (
    <div className="bg-slate-900 min-h-screen p-4 sm:p-8 font-[family-name:var(--font-geist-sans)]">
      <main className="max-w-7xl mx-auto h-[calc(100vh-200px)]">
        <div className="mb-8 mx-4">
          <h1 className="text-7xl font-bold bg-gradient-to-r from-blue-600 to-blue-400 bg-clip-text text-transparent font-mono">
            Watcher
          </h1>
          <p className="text-m text-blue-400/80 mt-2">
            Make smarter trades with AI market predictions and sentiment
            analysis of financial news
          </p>
        </div>
        <div className="h-px bg-gradient-to-r from-blue-600/20 via-blue-400/20 to-transparent mb-8" />
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full">
          <div className="lg:col-span-2 space-y-8">
            <StockChart />
            <PredictionPrice />
          </div>
          <div className="lg:col-span-1">
            <h2 className="mx-4 text-lg font-medium text-white mb-4 sticky top-0 bg-slate-900 z-10">
              Key News and Sentiment Analysis
            </h2>
            <div className="overflow-y-auto h-[calc(100vh-20rem)] [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-slate-700/50 hover:[&::-webkit-scrollbar-thumb]:bg-slate-700 [&::-webkit-scrollbar-thumb]:rounded-full">
              <NewsArticleList />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
