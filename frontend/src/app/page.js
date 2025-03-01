import { StockChart } from "@/components/StockChart";
import { PredictionPrice } from "@/components/PredictionPrice";
import { NewsArticle, NewsArticleList } from "@/components/NewsArticle";

export default function Home() {
  //const [stock, setStock] = React.useState("ALIBABA");

  return (
    <div className="bg-slate-900 min-h-screen p-4 sm:p-8 font-[family-name:var(--font-geist-sans)]">
      <h1 className="text-8xl font-bold text-blue-700 tracking-wider mb-6 pb-4 border-b-2 border-blue-900">
        Watcher
      </h1>
      <main className="max-w-7xl mx-auto h-[calc(100vh-200px)]">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 h-full">
          <div className="lg:col-span-2 space-y-8">
            <StockChart />
            <PredictionPrice />
          </div>
          <div className="lg:col-span-1 overflow-y-auto [&::-webkit-scrollbar]:w-2 [&::-webkit-scrollbar-track]:bg-transparent [&::-webkit-scrollbar-thumb]:bg-slate-700 [&::-webkit-scrollbar-thumb]:rounded-full">
            <NewsArticleList />
          </div>
        </div>
      </main>
    </div>
  );
}
