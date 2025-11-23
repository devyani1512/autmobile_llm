// import { useNavigate, useSearchParams } from "react-router-dom";
// import { Search, FileText, ArrowLeft } from "lucide-react";
// import { motion } from "framer-motion";
// import { Button } from "@/components/ui/button";
// import { Card } from "@/components/ui/card";

// const MechanicDashboard = () => {
//   const navigate = useNavigate();
//   const [searchParams] = useSearchParams();
//   const brand = searchParams.get("brand");
//   const model = searchParams.get("model");

//   const cardVariants = {
//     hidden: { opacity: 0, y: 30 },
//     visible: (i: number) => ({
//       opacity: 1,
//       y: 0,
//       transition: {
//         delay: i * 0.15,
//         duration: 0.5,
//         ease: "easeOut" as const
//       }
//     })
//   };

//   return (
//     <div className="min-h-screen bg-gradient-hero relative overflow-hidden">
//       {/* Background elements */}
//       <div className="absolute inset-0 opacity-20">
//         <div className="absolute top-1/3 left-1/3 w-96 h-96 bg-accent-mechanic/30 rounded-full blur-3xl animate-float" />
//         <div className="absolute bottom-1/3 right-1/3 w-96 h-96 bg-secondary/20 rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }} />
//       </div>

//       {/* Header */}
//       <header className="border-b border-border glass-card relative z-10">
//         <div className="container mx-auto px-4 py-6">
//           <div className="flex items-center justify-between">
//             <Button
//               variant="ghost"
//               onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
//               className="gap-2 hover:bg-accent-mechanic/10"
//             >
//               <ArrowLeft className="w-4 h-4" />
//               Change Mode
//             </Button>
//             <div className="text-sm text-muted-foreground">
//               Home <span className="text-foreground mx-2">›</span> {brand} {model} <span className="text-foreground mx-2">›</span> Mechanic Mode
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Main Content */}
//       <div className="container mx-auto px-4 py-16 relative z-10">
//         <motion.div
//           initial={{ opacity: 0, y: -30 }}
//           animate={{ opacity: 1, y: 0 }}
//           transition={{ duration: 0.8 }}
//           className="text-center mb-16"
//         >
//           <h1 className="text-5xl md:text-7xl font-bold mb-6">
//             <span className="text-gradient-gold uppercase tracking-wider">
//               Professional Workshop
//             </span>
//           </h1>
//           <p className="text-xl text-muted-foreground font-light">
//             Advanced diagnostic and reporting tools for professionals
//           </p>
//         </motion.div>

//         {/* Feature Cards */}
//         <div className="grid md:grid-cols-2 gap-12 max-w-6xl mx-auto">
//           {/* Diagnose Problem Card */}
//           <motion.div
//             custom={0}
//             variants={cardVariants}
//             initial="hidden"
//             animate="visible"
//             whileHover={{ y: -15, scale: 1.03 }}
//             onClick={() => navigate(`/chat?mode=mechanic&brand=${brand}&model=${model}`)}
//             className="cursor-pointer"
//           >
//             <Card className="relative overflow-hidden h-full p-12 glass-card border-2 border-transparent hover:border-accent-mechanic/50 transition-all duration-500 group">
//               <div className="absolute inset-0 bg-gradient-gold opacity-0 group-hover:opacity-15 transition-opacity duration-500" />
//               <div className="absolute inset-0 shadow-glow-gold opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
//               <div className="relative z-10 text-center">
//                 <div className="w-24 h-24 rounded-2xl bg-gradient-gold flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
//                   <Search className="w-12 h-12 text-background" />
//                 </div>
                
//                 <h3 className="text-4xl font-bold mb-6 uppercase tracking-wide">Advanced Diagnostics</h3>
//                 <p className="text-muted-foreground mb-8 leading-relaxed text-lg font-light">
//                   AI-powered troubleshooting with technical specifications and repair procedures. Access comprehensive diagnostic information and detailed repair instructions.
//                 </p>
                
//                 <Button className="w-full bg-gradient-gold hover:opacity-90 text-background font-bold py-6 text-lg uppercase">
//                   Start Diagnosis
//                 </Button>
//               </div>
//             </Card>
//           </motion.div>

//           {/* Generate Report Card */}
//           <motion.div
//             custom={1}
//             variants={cardVariants}
//             initial="hidden"
//             animate="visible"
//             whileHover={{ y: -15, scale: 1.03 }}
//             onClick={() => navigate(`/report-generation?brand=${brand}&model=${model}`)}
//             className="cursor-pointer"
//           >
//             <Card className="relative overflow-hidden h-full p-12 glass-card border-2 border-transparent hover:border-secondary/50 transition-all duration-500 group">
//               <div className="absolute inset-0 bg-gradient-silver opacity-0 group-hover:opacity-10 transition-opacity duration-500" />
//               <div className="absolute inset-0 shadow-glow-silver opacity-0 group-hover:opacity-100 transition-opacity duration-500" />
              
//               <div className="relative z-10 text-center">
//                 <div className="w-24 h-24 rounded-2xl bg-gradient-silver flex items-center justify-center mb-8 mx-auto group-hover:scale-110 transition-transform duration-500 shadow-elegant">
//                   <FileText className="w-12 h-12 text-background" />
//                 </div>
                
//                 <h3 className="text-4xl font-bold mb-6 uppercase tracking-wide">Service Reports</h3>
//                 <p className="text-muted-foreground mb-8 leading-relaxed text-lg font-light">
//                   Create comprehensive diagnostic and service reports with detailed analysis. Professional formatting ready for client delivery.
//                 </p>
                
//                 <Button variant="outline" className="w-full border-2 border-secondary hover:bg-secondary hover:text-background font-bold py-6 text-lg uppercase transition-all duration-300">
//                   Create Report
//                 </Button>
//               </div>
//             </Card>
//           </motion.div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default MechanicDashboard;
import { useNavigate, useSearchParams } from "react-router-dom";
import { Search, FileText, ArrowLeft } from "lucide-react";
import { motion } from "framer-motion";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";

const MechanicDashboard = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand");
  const model = searchParams.get("model");

  const cardVariants = {
    hidden: { opacity: 0, y: 30 },
    visible: (i: number) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.12,
        duration: 0.55,
        ease: "easeOut" as const
      }
    })
  };

  return (
    <div className="min-h-screen bg-black text-zinc-200 relative overflow-hidden" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>
      {/* Subtle silver gradvancedn background */}
      <div className="absolute inset-0 opacity-40 pointer-events-none">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_10%_20%,rgba(255,255,255,0.02),transparent),radial-gradient(circle_at_90%_80%,rgba(255,255,255,0.01),transparent)]" />
        <div className="absolute -left-40 -top-40 w-96 h-96 rounded-2xl bg-gradient-to-br from-zinc-800 to-zinc-900 opacity-20 blur-3xl transform rotate-12" />
        <div className="absolute -right-28 -bottom-28 w-96 h-96 rounded-2xl bg-gradient-to-bl from-zinc-700 to-zinc-900 opacity-12 blur-3xl transform -rotate-6" />
      </div>

      {/* Header */}
      <header className="relative z-10 border-b border-zinc-800 bg-transparent">
        <div className="contadvancedner mx-auto px-6 py-6 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Button
              variant="ghost"
              onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
              className="gap-2 hover:bg-zinc-800/40 text-zinc-200 rounded-none px-3 py-2"
            >
              <ArrowLeft className="w-4 h-4" />
              <span style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }} className="text-sm">Change Mode</span>
            </Button>

            <div className="ml-2 text-sm text-zinc-400" style={{ fontFamily: "Georgia, serif" }}>
              Home <span className="text-zinc-200 mx-2">›</span> {brand} {model} <span className="text-zinc-200 mx-2">›</span> Mechanic Mode
            </div>
          </div>

          <div className="text-right">
            <div className="text-xs text-zinc-400">Professional Suite</div>
            <div className="text-sm text-zinc-200 font-semibold" style={{ letterSpacing: '0.6px' }}>{brand || 'Brand'} · {model || 'Model'}</div>
          </div>
        </div>
      </header>

      {/* Madvancedn Content */}
      <div className="contadvancedner mx-auto px-6 py-16 relative z-10">
        <motion.div
          initial={{ opacity: 0, y: -18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8 }}
          className="text-center mb-14"
        >
          <h1 className="text-5xl md:text-6xl font-extrabold mb-4 leading-tight" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-zinc-200 to-zinc-400" style={{ WebkitBackgroundClip: 'text' }}>
              Professional Workshop
            </span>
          </h1>
          <p className="text-lg text-zinc-400 font-light max-w-3xl mx-auto" style={{ fontFamily: 'Georgia, serif' }}>
            Advanced‑augmented diagnostic intelligence engineered for high‑performance automotive environments — precise, predictive, and technician‑ready.
          </p>
        </motion.div>

        {/* Feature Cards */}
        <div className="grid md:grid-cols-2 gap-10 max-w-6xl mx-auto">
          {/* Diagnose Problem Card */}
          <motion.div
            custom={0}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -12, scale: 1.02 }}
            onClick={() => navigate(`/chat?mode=mechanic&brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-10 bg-gradient-to-b from-zinc-900/70 to-zinc-900/40 border border-zinc-800 transition-shadow duration-500 group rounded-none">
              <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-60 transition-opacity duration-700" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.03))' }} />

              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-lg bg-zinc-800/60 flex items-center justify-center mb-6 mx-auto shadow-[0_20px_40px_rgba(0,0,0,0.6)]">
                  <Search className="w-12 h-12 text-zinc-200" />
                </div>

                <h3 className="text-3xl font-semibold mb-4 uppercase tracking-wider" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>Advanced Diagnostics</h3>
                <p className="text-zinc-400 mb-8 leading-relaxed text-base font-light" style={{ fontFamily: 'Georgia, serif' }}>
                  Neural diagnostic reasoning that interprets symptoms, predicts fadvancedlures, and generates precision‑grade repadvancedr pathways tadvancedlored to the vehicle’s engineering profile.
                </p>

                <Button className="w-full bg-black hover:bg-zinc-900 text-zinc-200 font-semibold py-4 text-sm uppercase rounded-none border border-zinc-700 tracking-wider">
                  <span className="sr-only">Start Diagnosis</span>
                  <span style={{ color: '#ffffff' }}>Start Diagnosis</span>
                </Button>
              </div>
            </Card>
          </motion.div>

          {/* Generate Report Card */}
          <motion.div
            custom={1}
            variants={cardVariants}
            initial="hidden"
            animate="visible"
            whileHover={{ y: -12, scale: 1.02 }}
            onClick={() => navigate(`/report-generation?brand=${brand}&model=${model}`)}
            className="cursor-pointer"
          >
            <Card className="relative overflow-hidden h-full p-10 bg-gradient-to-b from-zinc-900/70 to-zinc-900/40 border border-zinc-800 transition-shadow duration-500 group rounded-none">
              <div className="absolute inset-0 pointer-events-none opacity-0 group-hover:opacity-60 transition-opacity duration-700" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.01), rgba(255,255,255,0.03))' }} />

              <div className="relative z-10 text-center">
                <div className="w-24 h-24 rounded-lg bg-zinc-800/60 flex items-center justify-center mb-6 mx-auto shadow-[0_20px_40px_rgba(0,0,0,0.6)]">
                  <FileText className="w-12 h-12 text-zinc-200" />
                </div>

                <h3 className="text-3xl font-semibold mb-4 uppercase tracking-wider" style={{ fontFamily: "'Playfadvancedr Display', Georgia, serif" }}>Service Reports</h3>
                <p className="text-zinc-400 mb-8 leading-relaxed text-base font-light" style={{ fontFamily: 'Georgia, serif' }}>
                  Generate Advanced‑structured service intelligence reports with layered insights, component‑level reasoning, and executive‑grade presentation formatting.
                </p>

                <Button variant="outline" className="w-full border-2 border-zinc-600 hover:bg-zinc-800/60 hover:text-zinc-200 font-bold py-4 text-sm uppercase transition-all duration-300 rounded-none">
                  Create Report
                </Button>
              </div>
            </Card>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default MechanicDashboard;
