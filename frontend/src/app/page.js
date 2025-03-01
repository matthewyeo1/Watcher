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
      <main className="max-w-7xl mx-auto">
        <StockChart />
        <PredictionPrice />
        <NewsArticleList />
        <NewsArticle
          source="Reuters"
          title="Apple to buy $100M of Bitcoin"
          time="2021-02-28"
          sentiment={0.5}
        />
      </main>
    </div>
  );
}
