import { useNavigate, useSearchParams } from "react-router-dom";
import { MessageSquare, BookOpen, AlertTriangle, ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const OwnerDashboard = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand");
  const model = searchParams.get("model");

  const cardVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.12,
        duration: 0.55,
        ease: [0.16, 1, 0.3, 1],
      },
    }),
  };

  return (
    <div className="min-h-screen bg-black text-zinc-200 relative overflow-hidden font-serif">
      
      <div className="absolute inset-0 pointer-events-none opacity-10 bg-[radial-gradient(circle_at_20%_30%,#ffffff10,transparent_60%),radial-gradient(circle_at_80%_70%,#ffffff08,transparent_60%)]" />

      
      <header className="border-b border-zinc-800 bg-black/60 backdrop-blur-sm relative z-10">
        <div className="container mx-auto px-6 py-6 flex items-center justify-between">
          <Button
            variant="ghost"
            onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
            className="gap-2 hover:bg-white/5 text-zinc-200"
          >
            <ArrowLeft className="w-4 h-4" />
            Change Mode
          </Button>

          <div className="text-sm text-zinc-400">
            Home <span className="text-zinc-300 mx-2">›</span> {brand} {model}
            <span className="text-zinc-300 mx-2">›</span> Owner Mode
          </div>
        </div>
      </header>

      
      <div className="container mx-auto px-6 py-20 relative z-10 text-center">
        <motion.h1
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-6xl md:text-7xl font-bold mb-6 tracking-tight"
        >
          <span className="text-zinc-100">Welcome,</span>
          <br />
          <span className="bg-clip-text text-transparent bg-gradient-to-r from-zinc-300 to-zinc-500 uppercase">
            Vehicle Owner
          </span>
        </motion.h1>

        <p className="text-xl text-zinc-400 font-light max-w-2xl mx-auto">
          Your personal automotive assistant — refined, precise, and engineered for clarity.
        </p>
      </div>

      
      <div className="container mx-auto px-6 pb-16 max-w-7xl grid md:grid-cols-3 gap-12">
        
        <motion.div
          custom={0}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          whileHover={{ y: -10, scale: 1.01 }}
          onClick={() => navigate(`/chat?mode=owner&brand=${brand}&model=${model}`)}
          className="cursor-pointer"
        >
          <Card className="p-10 bg-zinc-950/80 border border-zinc-800 hover:border-zinc-600 transition-all duration-300 rounded-none shadow-[0_0_40px_-20px_#ffffff20]">
            <div className="w-20 h-20 mx-auto mb-8 rounded-lg bg-gradient-to-b from-zinc-700 to-zinc-900 flex items-center justify-center shadow-inner">
              <MessageSquare className="w-10 h-10 text-white" />
            </div>

            <h3 className="text-2xl font-semibold mb-4 uppercase tracking-wide text-zinc-100 text-center">
              Ask Technical Questions
            </h3>
            <p className="text-zinc-400 mb-8 text-center leading-relaxed">
              Immediate expert-level answers for maintenance, features, and troubleshooting.
            </p>

            <Button className="w-full bg-zinc-800 hover:bg-zinc-700 text-white py-5 text-lg rounded-none font-semibold tracking-wide">
              Start Conversation
            </Button>
          </Card>
        </motion.div>

        
        <motion.div
          custom={1}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          whileHover={{ y: -10, scale: 1.01 }}
          onClick={() => navigate(`/vehicle-explorer?brand=${brand}&model=${model}`)}
          className="cursor-pointer"
        >
          <Card className="p-10 bg-zinc-950/80 border border-zinc-800 hover:border-zinc-600 transition-all duration-300 rounded-none shadow-[0_0_40px_-20px_#ffffff20]">
            <div className="w-20 h-20 mx-auto mb-8 rounded-lg bg-gradient-to-b from-zinc-600 to-zinc-800 flex items-center justify-center shadow-inner">
              <BookOpen className="w-10 h-10 text-white" />
            </div>

            <h3 className="text-2xl font-semibold mb-4 uppercase tracking-wide text-zinc-100 text-center">
              Explore Your Vehicle
            </h3>
            <p className="text-zinc-400 mb-8 text-center leading-relaxed">
              A refined, interactive guide to every major system and component.
            </p>

            <Button variant="outline" className="w-full border-2 border-zinc-600 hover:bg-zinc-700 hover:text-white py-5 text-lg rounded-none font-semibold tracking-wide">
              Learn More
            </Button>
          </Card>
        </motion.div>

        
        <motion.div
          custom={2}
          variants={cardVariants}
          initial="hidden"
          animate="visible"
          whileHover={{ y: -10, scale: 1.01 }}
          onClick={() => navigate(`/emergency?brand=${brand}&model=${model}`)}
          className="cursor-pointer"
        >
          <Card className="p-10 bg-zinc-950/80 border border-zinc-800 hover:border-red-600 transition-all duration-300 rounded-none shadow-[0_0_40px_-20px_#ff000020]">
            <div className="w-20 h-20 mx-auto mb-8 rounded-lg bg-gradient-to-b from-red-700 to-red-900 flex items-center justify-center shadow-inner">
              <AlertTriangle className="w-10 h-10 text-white" />
            </div>

            <h3 className="text-2xl font-semibold mb-4 uppercase tracking-wide text-zinc-100 text-center">
              Emergency Support
            </h3>
            <p className="text-zinc-400 mb-8 text-center leading-relaxed">
              Quick-access safety procedures and emergency troubleshooting.
            </p>

            <Button className="w-full bg-red-700 hover:bg-red-600 text-white py-5 text-lg rounded-none font-semibold tracking-wide">
              Access Emergency Guide
            </Button>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default OwnerDashboard;