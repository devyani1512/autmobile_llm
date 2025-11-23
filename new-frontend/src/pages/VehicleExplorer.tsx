// // import { useState } from "react";
// // import { useNavigate } from "react-router-dom";
// // import { ArrowLeft, Info } from "lucide-react";
// // import { motion, AnimatePresence } from "framer-motion";
// // import { Button } from "@/components/ui/button";
// // import { Card } from "@/components/ui/card";

// // interface Hotspot {
// //   id: string;
// //   x: number;
// //   y: number;
// //   label: string;
// //   description: string;
// //   tips: string[];
// // }

// // const hotspots: Hotspot[] = [
// //   {
// //     id: "engine",
// //     x: 30,
// //     y: 45,
// //     label: "Engine",
// //     description: "The heart of your vehicle, converting fuel into mechanical energy.",
// //     tips: ["Check oil levels monthly", "Regular maintenance every 5,000 miles", "Listen for unusual sounds"]
// //   },
// //   {
// //     id: "transmission",
// //     x: 45,
// //     y: 52,
// //     label: "Transmission",
// //     description: "Transfers power from the engine to the wheels.",
// //     tips: ["Check transmission fluid regularly", "Smooth shifting is key", "Service every 30,000 miles"]
// //   },
// //   {
// //     id: "brakes",
// //     x: 70,
// //     y: 60,
// //     label: "Brake System",
// //     description: "Critical safety system for stopping your vehicle.",
// //     tips: ["Replace pads when worn", "Check brake fluid monthly", "Listen for squeaking sounds"]
// //   },
// //   {
// //     id: "battery",
// //     x: 25,
// //     y: 35,
// //     label: "Battery",
// //     description: "Powers electrical systems and starts the engine.",
// //     tips: ["Clean terminals regularly", "Replace every 3-5 years", "Check voltage periodically"]
// //   }
// // ];

// // const VehicleExplorer = () => {
// //   const navigate = useNavigate();
// //   const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);

// //   return (
// //     <div className="min-h-screen bg-gradient-hero">
// //       {/* Header */}
// //       <header className="border-b border-border glass-card">
// //         <div className="container mx-auto px-4 py-6">
// //           <div className="flex items-center justify-between">
// //             <Button
// //               variant="ghost"
// //               onClick={() => navigate("/dashboard/owner")}
// //               className="gap-2"
// //             >
// //               <ArrowLeft className="w-4 h-4" />
// //               Back to Dashboard
// //             </Button>
// //             <h1 className="text-xl font-bold text-gradient-owner">Vehicle Explorer</h1>
// //             <div className="w-24" />
// //           </div>
// //         </div>
// //       </header>

// //       {/* Main Content */}
// //       <div className="container mx-auto px-4 py-12">
// //         <div className="grid md:grid-cols-2 gap-8 max-w-7xl mx-auto">
// //           {/* Vehicle Visualization */}
// //           <motion.div
// //             initial={{ opacity: 0, x: -30 }}
// //             animate={{ opacity: 1, x: 0 }}
// //             transition={{ duration: 0.6 }}
// //             className="relative"
// //           >
// //             <Card className="glass-card p-8">
// //               <h2 className="text-2xl font-bold mb-6">Interactive Vehicle Guide</h2>
              
// //               {/* Car SVG with hotspots */}
// //               <div className="relative w-full aspect-video bg-muted/20 rounded-lg overflow-hidden">
// //                 <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
// //                   {/* Car outline */}
// //                   <path
// //                     d="M 15 45 L 20 45 L 22 40 L 30 40 L 32 35 L 68 35 L 70 40 L 78 40 L 80 45 L 85 45 L 85 55 L 82 55 C 82 60, 78 65, 73 65 C 68 65, 64 60, 64 55 L 36 55 C 36 60, 32 65, 27 65 C 22 65, 18 60, 18 55 L 15 55 Z"
// //                     fill="hsl(0 0% 14%)"
// //                     stroke="hsl(210 100% 50%)"
// //                     strokeWidth="0.5"
// //                   />
                  
// //                   {/* Hotspots */}
// //                   {hotspots.map((hotspot) => (
// //                     <g key={hotspot.id}>
// //                       <motion.circle
// //                         cx={hotspot.x}
// //                         cy={hotspot.y}
// //                         r="2"
// //                         fill="hsl(210 100% 50%)"
// //                         className="cursor-pointer"
// //                         onClick={() => setSelectedHotspot(hotspot)}
// //                         whileHover={{ scale: 1.5 }}
// //                         animate={{
// //                           opacity: [0.6, 1, 0.6],
// //                           scale: [1, 1.2, 1]
// //                         }}
// //                         transition={{
// //                           duration: 2,
// //                           repeat: Infinity,
// //                           ease: "easeInOut"
// //                         }}
// //                       />
// //                       <circle
// //                         cx={hotspot.x}
// //                         cy={hotspot.y}
// //                         r="3"
// //                         fill="none"
// //                         stroke="hsl(210 100% 50%)"
// //                         strokeWidth="0.3"
// //                         opacity="0.5"
// //                         className="pointer-events-none"
// //                       />
// //                     </g>
// //                   ))}
// //                 </svg>
// //               </div>

// //               <p className="text-sm text-muted-foreground mt-4 text-center">
// //                 Click on the glowing points to learn more about each component
// //               </p>
// //             </Card>
// //           </motion.div>

// //           {/* Info Panel */}
// //           <motion.div
// //             initial={{ opacity: 0, x: 30 }}
// //             animate={{ opacity: 1, x: 0 }}
// //             transition={{ duration: 0.6 }}
// //           >
// //             <AnimatePresence mode="wait">
// //               {selectedHotspot ? (
// //                 <motion.div
// //                   key={selectedHotspot.id}
// //                   initial={{ opacity: 0, y: 20 }}
// //                   animate={{ opacity: 1, y: 0 }}
// //                   exit={{ opacity: 0, y: -20 }}
// //                   transition={{ duration: 0.3 }}
// //                 >
// //                   <Card className="glass-card p-8 border-2 border-accent-owner/50 shadow-glow-owner">
// //                     <div className="flex items-start gap-4 mb-6">
// //                       <div className="w-12 h-12 rounded-lg bg-gradient-owner flex items-center justify-center">
// //                         <Info className="w-6 h-6 text-white" />
// //                       </div>
// //                       <div>
// //                         <h3 className="text-2xl font-bold text-gradient-owner">{selectedHotspot.label}</h3>
// //                         <p className="text-muted-foreground mt-2">{selectedHotspot.description}</p>
// //                       </div>
// //                     </div>

// //                     <div className="mb-6">
// //                       <h4 className="text-lg font-semibold mb-3">Maintenance Tips</h4>
// //                       <ul className="space-y-2">
// //                         {selectedHotspot.tips.map((tip, index) => (
// //                           <li key={index} className="flex items-start gap-2">
// //                             <div className="w-1.5 h-1.5 rounded-full bg-accent-owner mt-2" />
// //                             <span className="text-muted-foreground">{tip}</span>
// //                           </li>
// //                         ))}
// //                       </ul>
// //                     </div>

// //                     <Button
// //                       onClick={() => navigate("/chat?mode=owner&component=" + selectedHotspot.id)}
// //                       className="w-full bg-gradient-owner hover:opacity-90 text-white"
// //                     >
// //                       Ask AI About This Component
// //                     </Button>
// //                   </Card>
// //                 </motion.div>
// //               ) : (
// //                 <motion.div
// //                   initial={{ opacity: 0 }}
// //                   animate={{ opacity: 1 }}
// //                   exit={{ opacity: 0 }}
// //                 >
// //                   <Card className="glass-card p-8 h-full flex items-center justify-center">
// //                     <div className="text-center">
// //                       <div className="w-16 h-16 rounded-full bg-gradient-owner/20 flex items-center justify-center mx-auto mb-4">
// //                         <Info className="w-8 h-8 text-accent-owner" />
// //                       </div>
// //                       <p className="text-muted-foreground">
// //                         Select a component to learn more
// //                       </p>
// //                     </div>
// //                   </Card>
// //                 </motion.div>
// //               )}
// //             </AnimatePresence>
// //           </motion.div>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default VehicleExplorer;
// import { useState } from "react";
// import { useNavigate, useSearchParams } from "react-router-dom";
// import { ArrowLeft, Loader2, BookOpen, Sparkles, ChevronRight } from "lucide-react";
// import { motion, AnimatePresence } from "framer-motion";

// interface Hotspot {
//   id: string;
//   x: number;
//   y: number;
//   label: string;
//   description: string;
//   tips: string[];
// }

// interface ManualData {
//   answer_text: string;
//   images?: string[];
//   tables?: string[];
// }

// const hotspots: Hotspot[] = [
//   {
//     id: "engine",
//     x: 30,
//     y: 45,
//     label: "Engine",
//     description: "The heart of your vehicle, converting fuel into mechanical energy.",
//     tips: ["Check oil levels monthly", "Regular maintenance every 5,000 miles", "Listen for unusual sounds"]
//   },
//   {
//     id: "transmission",
//     x: 45,
//     y: 52,
//     label: "Transmission",
//     description: "Transfers power from the engine to the wheels.",
//     tips: ["Check transmission fluid regularly", "Smooth shifting is key", "Service every 30,000 miles"]
//   },
//   {
//     id: "brakes",
//     x: 70,
//     y: 60,
//     label: "Brake System",
//     description: "Critical safety system for stopping your vehicle.",
//     tips: ["Replace pads when worn", "Check brake fluid monthly", "Listen for squeaking sounds"]
//   },
//   {
//     id: "battery",
//     x: 25,
//     y: 35,
//     label: "Battery",
//     description: "Powers electrical systems and starts the engine.",
//     tips: ["Clean terminals regularly", "Replace every 3-5 years", "Check voltage periodically"]
//   }
// ];

// const VehicleExplorer = () => {
//   const navigate = useNavigate();
//   const [searchParams] = useSearchParams();
//   const brand = searchParams.get("brand") || "";
//   const model = searchParams.get("model") || "";
  
//   const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);
//   const [manualData, setManualData] = useState<ManualData | null>(null);
//   const [isLoadingManual, setIsLoadingManual] = useState(false);
//   const [manualError, setManualError] = useState<string | null>(null);

//   // Format vehicle name
//   const vehicleName = `${brand.charAt(0).toUpperCase() + brand.slice(1)} ${model.split('-').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}`;

//   // Fetch manual data when a hotspot is selected
//   const fetchManualData = async (componentId: string, componentName: string) => {
//     setIsLoadingManual(true);
//     setManualError(null);
    
//     try {
//       const query = `What does the manual say about the ${componentName} in this vehicle? Include maintenance procedures, specifications, and important safety information.`;
      
//       console.log('🔍 Fetching manual data:', { query, brand, model });
      
//       const response = await fetch("http://localhost:8000/api/ask", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           query: query,
//           manufacturer: brand,
//           model: model,
//           mode: "owner"
//         }),
//       });

//       console.log('📡 Response status:', response.status);

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }

//       const data = await response.json();
//       console.log('✅ Manual data received:', data);
      
//       if (data.answer) {
//         setManualData(data.answer);
//       } else {
//         throw new Error('No answer data received');
//       }
//     } catch (error) {
//       console.error("❌ Error fetching manual data:", error);
//       setManualError("Could not load manual information. Please ensure the backend is running.");
//     } finally {
//       setIsLoadingManual(false);
//     }
//   };

//   // Handle hotspot selection
//   const handleHotspotClick = (hotspot: Hotspot) => {
//     setSelectedHotspot(hotspot);
//     setManualData(null);
//     fetchManualData(hotspot.id, hotspot.label);
//   };

//   return (
//     <div className="min-h-screen bg-black text-white relative overflow-hidden">
//       {/* Premium Background */}
//       <div className="fixed inset-0 z-0">
//         <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900 to-black" />
//         <div className="absolute inset-0 opacity-20">
//           <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl" />
//           <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl" />
//         </div>
        
//         {/* Animated lines */}
//         <div className="absolute inset-0 opacity-10">
//           {[...Array(20)].map((_, i) => (
//             <motion.div
//               key={i}
//               className="absolute w-px h-full bg-gradient-to-b from-transparent via-cyan-500 to-transparent"
//               style={{ left: `${i * 5}%` }}
//               animate={{
//                 opacity: [0.1, 0.3, 0.1],
//                 scaleY: [0.8, 1, 0.8],
//               }}
//               transition={{
//                 duration: 3 + Math.random() * 2,
//                 repeat: Infinity,
//                 delay: Math.random() * 2,
//               }}
//             />
//           ))}
//         </div>
//       </div>

//       {/* Luxury Header */}
//       <header className="relative z-50 border-b border-white/5 bg-black/40 backdrop-blur-xl">
//         <div className="max-w-7xl mx-auto px-8 py-6">
//           <div className="flex items-center justify-between">
//             <button
//               onClick={() => navigate(`/dashboard/owner?brand=${brand}&model=${model}`)}
//               className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors group"
//             >
//               <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
//               <span className="text-sm tracking-wider uppercase">Dashboard</span>
//             </button>
            
//             <div className="flex items-center gap-3">
//               <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-white/10">
//                 <Sparkles className="w-4 h-4 text-cyan-400" />
//                 <span className="text-sm font-light tracking-wider">{vehicleName}</span>
//               </div>
//             </div>
//           </div>
//         </div>
//       </header>

//       {/* Main Content */}
//       <div className="relative z-10 max-w-7xl mx-auto px-8 py-16">
        
//         {/* Title Section */}
//         <motion.div
//           initial={{ opacity: 0, y: 20 }}
//           animate={{ opacity: 1, y: 0 }}
//           className="text-center mb-16"
//         >
//           <h1 className="text-5xl md:text-6xl font-thin tracking-tight mb-4">
//             Vehicle <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">Explorer</span>
//           </h1>
//           <p className="text-gray-400 text-lg font-light">
//             Interactive component guide with manual integration
//           </p>
//         </motion.div>

//         {/* Two Column Layout */}
//         <div className="grid lg:grid-cols-2 gap-8">
          
//           {/* Left: Vehicle Visualization */}
//           <motion.div
//             initial={{ opacity: 0, x: -30 }}
//             animate={{ opacity: 1, x: 0 }}
//             transition={{ duration: 0.6 }}
//           >
//             <div className="relative bg-gradient-to-br from-gray-900/50 to-black/50 backdrop-blur-xl border border-white/10 rounded-2xl p-8 hover:border-cyan-500/30 transition-all duration-500">
//               {/* Glow effect */}
//               <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-cyan-500/5 to-blue-500/5 opacity-0 hover:opacity-100 transition-opacity duration-500" />
              
//               <div className="relative z-10">
//                 <h2 className="text-2xl font-light mb-6 tracking-wide">Component Map</h2>
                
//                 {/* Car SVG */}
//                 <div className="relative w-full aspect-video bg-black/30 rounded-xl overflow-hidden border border-white/5">
//                   <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRy="xMidYMid meet">
//                     {/* Car outline */}
//                     <defs>
//                       <linearGradient id="carGradient" x1="0%" y1="0%" x2="100%" y2="100%">
//                         <stop offset="0%" style={{ stopColor: '#1e293b', stopOpacity: 1 }} />
//                         <stop offset="100%" style={{ stopColor: '#0f172a', stopOpacity: 1 }} />
//                       </linearGradient>
//                     </defs>
                    
//                     <path
//                       d="M 15 45 L 20 45 L 22 40 L 30 40 L 32 35 L 68 35 L 70 40 L 78 40 L 80 45 L 85 45 L 85 55 L 82 55 C 82 60, 78 65, 73 65 C 68 65, 64 60, 64 55 L 36 55 C 36 60, 32 65, 27 65 C 22 65, 18 60, 18 55 L 15 55 Z"
//                       fill="url(#carGradient)"
//                       stroke="#0ea5e9"
//                       strokeWidth="0.3"
//                       opacity="0.8"
//                     />
                    
//                     {/* Hotspots */}
//                     {hotspots.map((hotspot) => (
//                       <g key={hotspot.id}>
//                         <motion.circle
//                           cx={hotspot.x}
//                           cy={hotspot.y}
//                           r="2.5"
//                           fill="#0ea5e9"
//                           className="cursor-pointer"
//                           onClick={() => handleHotspotClick(hotspot)}
//                           whileHover={{ scale: 1.5 }}
//                           animate={{
//                             opacity: selectedHotspot?.id === hotspot.id ? 1 : [0.5, 1, 0.5],
//                             scale: selectedHotspot?.id === hotspot.id ? 1.3 : [1, 1.2, 1]
//                           }}
//                           transition={{
//                             duration: 2,
//                             repeat: Infinity,
//                             ease: "easeInOut"
//                           }}
//                         />
//                         {/* Glow ring */}
//                         <circle
//                           cx={hotspot.x}
//                           cy={hotspot.y}
//                           r="4"
//                           fill="none"
//                           stroke="#06b6d4"
//                           strokeWidth="0.2"
//                           opacity="0.3"
//                           className="pointer-events-none"
//                         />
//                       </g>
//                     ))}
//                   </svg>
                  
//                   {/* Tech grid overlay */}
//                   <div className="absolute inset-0 opacity-5 pointer-events-none" style={{
//                     backgroundImage: 'linear-gradient(#0ea5e9 1px, transparent 1px), linear-gradient(90deg, #0ea5e9 1px, transparent 1px)',
//                     backgroundSize: '20px 20px'
//                   }} />
//                 </div>

//                 <p className="text-sm text-gray-500 mt-4 text-center font-light">
//                   Click glowing points to explore components
//                 </p>
//               </div>
//             </div>
//           </motion.div>

//           {/* Right: Component Details */}
//           <motion.div
//             initial={{ opacity: 0, x: 30 }}
//             animate={{ opacity: 1, x: 0 }}
//             transition={{ duration: 0.6 }}
//           >
//             <AnimatePresence mode="wait">
//               {selectedHotspot ? (
//                 <motion.div
//                   key={selectedHotspot.id}
//                   initial={{ opacity: 0, y: 20 }}
//                   animate={{ opacity: 1, y: 0 }}
//                   exit={{ opacity: 0, y: -20 }}
//                   className="relative bg-gradient-to-br from-gray-900/50 to-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-8 shadow-2xl shadow-cyan-500/10"
//                 >
//                   {/* Inner glow */}
//                   <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-cyan-500/10 to-blue-500/10" />
                  
//                   <div className="relative z-10">
//                     {/* Component Header */}
//                     <div className="mb-8">
//                       <div className="flex items-center gap-3 mb-3">
//                         <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
//                           <BookOpen className="w-6 h-6 text-white" />
//                         </div>
//                         <div>
//                           <h3 className="text-3xl font-light tracking-tight">{selectedHotspot.label}</h3>
//                           <p className="text-sm text-gray-400 mt-1">{selectedHotspot.description}</p>
//                         </div>
//                       </div>
//                       <div className="h-px w-20 bg-gradient-to-r from-cyan-500 to-transparent" />
//                     </div>

//                     {/* Quick Tips */}
//                     <div className="mb-8">
//                       <h4 className="text-sm uppercase tracking-widest text-gray-500 mb-4">Quick Reference</h4>
//                       <div className="space-y-2">
//                         {selectedHotspot.tips.map((tip, index) => (
//                           <div key={index} className="flex items-start gap-3 text-gray-400 text-sm">
//                             <div className="w-1 h-1 rounded-full bg-cyan-500 mt-2" />
//                             <span className="font-light">{tip}</span>
//                           </div>
//                         ))}
//                       </div>
//                     </div>

//                     {/* Manual Information */}
//                     <div className="border-t border-white/5 pt-6">
//                       <div className="flex items-center gap-2 mb-4">
//                         <Sparkles className="w-4 h-4 text-cyan-400" />
//                         <h4 className="text-sm uppercase tracking-widest text-gray-400">Official Manual</h4>
//                       </div>
                      
//                       {isLoadingManual ? (
//                         <div className="flex items-center gap-3 text-gray-500 py-8">
//                           <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
//                           <span className="font-light">Retrieving manual data...</span>
//                         </div>
//                       ) : manualError ? (
//                         <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg p-4">
//                           {manualError}
//                         </div>
//                       ) : manualData ? (
//                         <div>
//                           <p className="text-gray-300 leading-relaxed mb-6 whitespace-pre-wrap font-light">
//                             {manualData.answer_text}
//                           </p>
                          
//                           {/* Images */}
//                           {manualData.images && manualData.images.length > 0 && (
//                             <div className="grid grid-cols-2 gap-3 mb-6">
//                               {manualData.images.map((img, i) => (
//                                 <div key={i} className="relative rounded-lg overflow-hidden border border-white/10">
//                                   <img
//                                     src={img}
//                                     alt={`Manual diagram ${i + 1}`}
//                                     className="w-full h-auto"
//                                   />
//                                 </div>
//                               ))}
//                             </div>
//                           )}
                          
//                           {/* Tables */}
//                           {manualData.tables && manualData.tables.length > 0 && (
//                             <div className="space-y-3 mb-6">
//                               {manualData.tables.map((table, i) => (
//                                 <div key={i} className="p-4 bg-black/40 rounded-lg text-xs font-mono text-gray-400 border border-white/5">
//                                   {table}
//                                 </div>
//                               ))}
//                             </div>
//                           )}
//                         </div>
//                       ) : (
//                         <p className="text-gray-500 text-sm font-light">
//                           Manual information will appear here
//                         </p>
//                       )}
//                     </div>

//                     {/* Action Button */}
//                     <button
//                       onClick={() => navigate(`/chat?mode=owner&component=${selectedHotspot.id}&brand=${brand}&model=${model}`)}
//                       className="w-full mt-8 py-4 px-6 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-light tracking-wide transition-all duration-300 flex items-center justify-between group"
//                     >
//                       <span>Ask AI About This Component</span>
//                       <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
//                     </button>
//                   </div>
//                 </motion.div>
//               ) : (
//                 <motion.div
//                   initial={{ opacity: 0 }}
//                   animate={{ opacity: 1 }}
//                   exit={{ opacity: 0 }}
//                   className="relative bg-gradient-to-br from-gray-900/30 to-black/30 backdrop-blur-xl border border-white/5 rounded-2xl p-16 h-full flex items-center justify-center"
//                 >
//                   <div className="text-center">
//                     <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/20 flex items-center justify-center mx-auto mb-6">
//                       <BookOpen className="w-10 h-10 text-cyan-400/50" />
//                     </div>
//                     <p className="text-gray-400 text-lg font-light mb-2">
//                       Select a Component
//                     </p>
//                     <p className="text-gray-600 text-sm font-light">
//                       Click any glowing point to view details
//                     </p>
//                   </div>
//                 </motion.div>
//               )}
//             </AnimatePresence>
//           </motion.div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default VehicleExplorer;
import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, Loader2, BookOpen, Sparkles, ChevronRight, Gauge, Zap, Shield, Droplet, Wind, Battery } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Hotspot {
  id: string;
  x: number;
  y: number;
  label: string;
  description: string;
  tips: string[];
}

interface ManualData {
  answer_text: string;
  images?: string[];
  tables?: string[];
}

interface CarFeature {
  icon: any;
  label: string;
  value: string;
  color: string;
}

interface VehicleSpecs {
  topSpeed: string;
  acceleration: string;
  fuelEconomy: string;
  battery: string;
  safetyRating: string;
  emissions: string;
}

const hotspots: Hotspot[] = [
  {
    id: "engine",
    x: 30,
    y: 45,
    label: "Engine",
    description: "The heart of your vehicle, converting fuel into mechanical energy.",
    tips: ["Check oil levels monthly", "Regular maintenance every 5,000 miles", "Listen for unusual sounds"]
  },
  {
    id: "transmission",
    x: 45,
    y: 52,
    label: "Transmission",
    description: "Transfers power from the engine to the wheels.",
    tips: ["Check transmission fluid regularly", "Smooth shifting is key", "Service every 30,000 miles"]
  },
  {
    id: "brakes",
    x: 70,
    y: 60,
    label: "Brake System",
    description: "Critical safety system for stopping your vehicle.",
    tips: ["Replace pads when worn", "Check brake fluid monthly", "Listen for squeaking sounds"]
  },
  {
    id: "battery",
    x: 25,
    y: 35,
    label: "Battery",
    description: "Powers electrical systems and starts the engine.",
    tips: ["Clean terminals regularly", "Replace every 3-5 years", "Check voltage periodically"]
  }
];

const VehicleExplorer = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const brand = searchParams.get("brand") || "";
  const model = searchParams.get("model") || "";
  
  const [selectedHotspot, setSelectedHotspot] = useState<Hotspot | null>(null);
  const [manualData, setManualData] = useState<ManualData | null>(null);
  const [isLoadingManual, setIsLoadingManual] = useState(false);
  const [manualError, setManualError] = useState<string | null>(null);
  
  // New states for vehicle specs
  const [vehicleSpecs, setVehicleSpecs] = useState<VehicleSpecs | null>(null);
  const [isLoadingSpecs, setIsLoadingSpecs] = useState(true);

  const vehicleName = `${brand.charAt(0).toUpperCase() + brand.slice(1)} ${model.split('-').map((w: string) => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}`;

  // Fetch vehicle specifications from manual on component mount
  useEffect(() => {
    fetchVehicleSpecs();
  }, [brand, model]);

  const fetchVehicleSpecs = async () => {
    setIsLoadingSpecs(true);
    try {
      const specsQuery = `Extract the following specifications from the vehicle manual:
1. Top speed or maximum speed
2. 0-60 mph acceleration time (or 0-100 km/h)
3. Fuel economy (MPG or L/100km)
4. Battery type and voltage
5. Safety rating if mentioned
6. Emissions standard (Euro 6, etc.)

Provide ONLY the values in a clear format. If a specification is not found, write "Not specified".`;

      const response = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: specsQuery,
          manufacturer: brand,
          model: model,
          mode: "owner"
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      
      if (data.answer && data.answer.answer_text) {
        // Parse the response to extract specs
        const text = data.answer.answer_text;
        const specs = parseSpecsFromText(text);
        setVehicleSpecs(specs);
      } else {
        // Fallback to default values
        setVehicleSpecs(getDefaultSpecs());
      }
    } catch (error) {
      console.error("Error fetching vehicle specs:", error);
      // Use default specs on error
      setVehicleSpecs(getDefaultSpecs());
    } finally {
      setIsLoadingSpecs(false);
    }
  };

  const parseSpecsFromText = (text: string): VehicleSpecs => {
    // Simple parsing logic - can be enhanced based on your manual format
    const lines = text.split('\n');
    const specs: VehicleSpecs = {
      topSpeed: "Not specified",
      acceleration: "Not specified",
      fuelEconomy: "Not specified",
      battery: "Not specified",
      safetyRating: "Not specified",
      emissions: "Not specified"
    };

    lines.forEach(line => {
      const lowerLine = line.toLowerCase();
      if (lowerLine.includes('top speed') || lowerLine.includes('maximum speed')) {
        const match = line.match(/(\d+)\s*(mph|km\/h)/i);
        if (match) specs.topSpeed = `${match[1]} ${match[2]}`;
      }
      if (lowerLine.includes('0-60') || lowerLine.includes('acceleration')) {
        const match = line.match(/(\d+\.?\d*)\s*(sec|s|seconds)/i);
        if (match) specs.acceleration = `${match[1]} sec`;
      }
      if (lowerLine.includes('fuel economy') || lowerLine.includes('mpg')) {
        const match = line.match(/(\d+\.?\d*)\s*(mpg|l\/100km)/i);
        if (match) specs.fuelEconomy = `${match[1]} ${match[2]}`;
      }
      if (lowerLine.includes('battery')) {
        const match = line.match(/(\d+V?\s*\w+)/i);
        if (match) specs.battery = match[1];
      }
      if (lowerLine.includes('safety rating')) {
        const match = line.match(/(\d+)\s*star/i);
        if (match) specs.safetyRating = `${match[1]} Stars`;
      }
      if (lowerLine.includes('emission')) {
        const match = line.match(/(Euro\s*\d+|ULEV|LEV)/i);
        if (match) specs.emissions = match[1];
      }
    });

    return specs;
  };

  const getDefaultSpecs = (): VehicleSpecs => {
    return {
      topSpeed: "155 mph",
      acceleration: "6.8 sec",
      fuelEconomy: "32 mpg",
      battery: "12V AGM",
      safetyRating: "5 Stars",
      emissions: "Euro 6"
    };
  };

  const carFeatures: CarFeature[] = vehicleSpecs ? [
    { icon: Gauge, label: "Top Speed", value: vehicleSpecs.topSpeed, color: "from-blue-500 to-cyan-500" },
    { icon: Zap, label: "0-60 mph", value: vehicleSpecs.acceleration, color: "from-purple-500 to-pink-500" },
    { icon: Droplet, label: "Fuel Economy", value: vehicleSpecs.fuelEconomy, color: "from-green-500 to-emerald-500" },
    { icon: Battery, label: "Battery", value: vehicleSpecs.battery, color: "from-orange-500 to-red-500" },
    { icon: Shield, label: "Safety Rating", value: vehicleSpecs.safetyRating, color: "from-yellow-500 to-amber-500" },
    { icon: Wind, label: "Emissions", value: vehicleSpecs.emissions, color: "from-teal-500 to-cyan-500" },
  ] : [];

  const fetchManualData = async (componentId: string, componentName: string) => {
    setIsLoadingManual(true);
    setManualError(null);
    
    try {
      // Enhanced query to specifically request images and diagrams
      const query = `What does the manual say about the ${componentName} in this vehicle? 
Please include:
1. Detailed component information and specifications
2. Maintenance procedures
3. Safety information
4. ANY diagrams, images, or illustrations related to this component
5. Technical tables or charts if available

Please ensure to include all visual aids (diagrams, photos, illustrations) that show this component.`;
      
      console.log('🔍 Fetching manual data with images:', { query, brand, model });
      
      const response = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query,
          manufacturer: brand,
          model: model,
          mode: "owner",
          component: componentId  // Pass component info to backend
        }),
      });

      console.log('📡 Response status:', response.status);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ Manual data received:', data);
      
      if (data.answer) {
        setManualData(data.answer);
      } else {
        throw new Error('No answer data received');
      }
    } catch (error) {
      console.error("❌ Error fetching manual data:", error);
      setManualError("Could not load manual information. Please ensure the backend is running.");
    } finally {
      setIsLoadingManual(false);
    }
  };

  const handleHotspotClick = (hotspot: Hotspot) => {
    setSelectedHotspot(hotspot);
    setManualData(null);
    fetchManualData(hotspot.id, hotspot.label);
  };

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Premium Background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900 to-black" />
        <div className="absolute inset-0 opacity-20">
          <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-blue-600/20 rounded-full blur-3xl" />
          <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-600/20 rounded-full blur-3xl" />
        </div>
        
        <div className="absolute inset-0 opacity-10">
          {[...Array(20)].map((_, i) => (
            <motion.div
              key={i}
              className="absolute w-px h-full bg-gradient-to-b from-transparent via-cyan-500 to-transparent"
              style={{ left: `${i * 5}%` }}
              animate={{
                opacity: [0.1, 0.3, 0.1],
                scaleY: [0.8, 1, 0.8],
              }}
              transition={{
                duration: 3 + Math.random() * 2,
                repeat: Infinity,
                delay: Math.random() * 2,
              }}
            />
          ))}
        </div>
      </div>

      {/* Header */}
      <header className="relative z-50 border-b border-white/5 bg-black/40 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-8 py-6">
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate(`/dashboard/owner?brand=${brand}&model=${model}`)}
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors group"
            >
              <ArrowLeft className="w-5 h-5 group-hover:-translate-x-1 transition-transform" />
              <span className="text-sm tracking-wider uppercase">Dashboard</span>
            </button>
            
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-gradient-to-r from-blue-600/20 to-purple-600/20 border border-white/10">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-light tracking-wider">{vehicleName}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="relative z-10 max-w-7xl mx-auto px-8 py-16">
        
        {/* Title Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="text-center mb-12"
        >
          <h1 className="text-5xl md:text-6xl font-thin tracking-tight mb-4">
            Vehicle <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-500">Explorer</span>
          </h1>
          <p className="text-gray-400 text-lg font-light">
            Interactive component guide with manual integration
          </p>
        </motion.div>

        {/* Car Features Grid */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-16"
        >
          <div className="bg-gradient-to-br from-gray-900/50 to-black/50 backdrop-blur-xl border border-white/10 rounded-2xl p-8">
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                <Gauge className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-2xl font-light tracking-wide">Vehicle Specifications</h2>
              {isLoadingSpecs && (
                <Loader2 className="w-5 h-5 animate-spin text-cyan-400 ml-2" />
              )}
            </div>
            
            {isLoadingSpecs ? (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {[...Array(6)].map((_, idx) => (
                  <div key={idx} className="bg-black/40 border border-white/10 rounded-xl p-4 h-32 animate-pulse">
                    <div className="w-12 h-12 bg-gray-700 rounded-lg mb-3 mx-auto" />
                    <div className="h-3 bg-gray-700 rounded mb-2" />
                    <div className="h-4 bg-gray-700 rounded w-2/3 mx-auto" />
                  </div>
                ))}
              </div>
            ) : (
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
                {carFeatures.map((feature, idx) => (
                  <motion.div
                    key={idx}
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    transition={{ delay: 0.3 + idx * 0.1 }}
                    className="relative group"
                  >
                    <div className="bg-black/40 border border-white/10 rounded-xl p-4 hover:border-cyan-500/30 transition-all duration-300 hover:transform hover:scale-105">
                      <div className={`w-12 h-12 rounded-lg bg-gradient-to-br ${feature.color} flex items-center justify-center mb-3 mx-auto`}>
                        <feature.icon className="w-6 h-6 text-white" />
                      </div>
                      <p className="text-xs text-gray-400 text-center mb-1">{feature.label}</p>
                      <p className="text-lg font-semibold text-center text-white">{feature.value}</p>
                    </div>
                    
                    <div className={`absolute inset-0 rounded-xl bg-gradient-to-br ${feature.color} opacity-0 group-hover:opacity-20 transition-opacity duration-300 blur-xl -z-10`} />
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </motion.div>

        {/* Two Column Layout */}
        <div className="grid lg:grid-cols-2 gap-8">
          
          {/* Left: Vehicle Visualization */}
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <div className="relative bg-gradient-to-br from-gray-900/50 to-black/50 backdrop-blur-xl border border-white/10 rounded-2xl p-8 hover:border-cyan-500/30 transition-all duration-500">
              <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-cyan-500/5 to-blue-500/5 opacity-0 hover:opacity-100 transition-opacity duration-500" />
              
              <div className="relative z-10">
                <h2 className="text-2xl font-light mb-6 tracking-wide">Component Map</h2>
                
                <div className="relative w-full aspect-video bg-black/30 rounded-xl overflow-hidden border border-white/5">
                  <svg className="w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
                    <defs>
                      <linearGradient id="carGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" style={{ stopColor: '#1e293b', stopOpacity: 1 }} />
                        <stop offset="100%" style={{ stopColor: '#0f172a', stopOpacity: 1 }} />
                      </linearGradient>
                    </defs>
                    
                    <path
                      d="M 15 45 L 20 45 L 22 40 L 30 40 L 32 35 L 68 35 L 70 40 L 78 40 L 80 45 L 85 45 L 85 55 L 82 55 C 82 60, 78 65, 73 65 C 68 65, 64 60, 64 55 L 36 55 C 36 60, 32 65, 27 65 C 22 65, 18 60, 18 55 L 15 55 Z"
                      fill="url(#carGradient)"
                      stroke="#0ea5e9"
                      strokeWidth="0.3"
                      opacity="0.8"
                    />
                    
                    {hotspots.map((hotspot) => (
                      <g key={hotspot.id}>
                        <motion.circle
                          cx={hotspot.x}
                          cy={hotspot.y}
                          r="2.5"
                          fill="#0ea5e9"
                          className="cursor-pointer"
                          onClick={() => handleHotspotClick(hotspot)}
                          whileHover={{ scale: 1.5 }}
                          animate={{
                            opacity: selectedHotspot?.id === hotspot.id ? 1 : [0.5, 1, 0.5],
                            scale: selectedHotspot?.id === hotspot.id ? 1.3 : [1, 1.2, 1]
                          }}
                          transition={{
                            duration: 2,
                            repeat: Infinity,
                            ease: "easeInOut"
                          }}
                        />
                        <circle
                          cx={hotspot.x}
                          cy={hotspot.y}
                          r="4"
                          fill="none"
                          stroke="#06b6d4"
                          strokeWidth="0.2"
                          opacity="0.3"
                          className="pointer-events-none"
                        />
                      </g>
                    ))}
                  </svg>
                  
                  <div className="absolute inset-0 opacity-5 pointer-events-none" style={{
                    backgroundImage: 'linear-gradient(#0ea5e9 1px, transparent 1px), linear-gradient(90deg, #0ea5e9 1px, transparent 1px)',
                    backgroundSize: '20px 20px'
                  }} />
                </div>

                <p className="text-sm text-gray-500 mt-4 text-center font-light">
                  Click glowing points to explore components
                </p>
              </div>
            </div>
          </motion.div>

          {/* Right: Component Details */}
          <motion.div
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
          >
            <AnimatePresence mode="wait">
              {selectedHotspot ? (
                <motion.div
                  key={selectedHotspot.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="relative bg-gradient-to-br from-gray-900/50 to-black/50 backdrop-blur-xl border border-cyan-500/30 rounded-2xl p-8 shadow-2xl shadow-cyan-500/10"
                >
                  <div className="absolute inset-0 rounded-2xl bg-gradient-to-br from-cyan-500/10 to-blue-500/10" />
                  
                  <div className="relative z-10">
                    <div className="mb-8">
                      <div className="flex items-center gap-3 mb-3">
                        <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center">
                          <BookOpen className="w-6 h-6 text-white" />
                        </div>
                        <div>
                          <h3 className="text-3xl font-light tracking-tight">{selectedHotspot.label}</h3>
                          <p className="text-sm text-gray-400 mt-1">{selectedHotspot.description}</p>
                        </div>
                      </div>
                      <div className="h-px w-20 bg-gradient-to-r from-cyan-500 to-transparent" />
                    </div>

                    <div className="mb-8">
                      <h4 className="text-sm uppercase tracking-widest text-gray-500 mb-4">Quick Reference</h4>
                      <div className="space-y-2">
                        {selectedHotspot.tips.map((tip, index) => (
                          <div key={index} className="flex items-start gap-3 text-gray-400 text-sm">
                            <div className="w-1 h-1 rounded-full bg-cyan-500 mt-2" />
                            <span className="font-light">{tip}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="border-t border-white/5 pt-6">
                      <div className="flex items-center gap-2 mb-4">
                        <Sparkles className="w-4 h-4 text-cyan-400" />
                        <h4 className="text-sm uppercase tracking-widest text-gray-400">Official Manual</h4>
                      </div>
                      
                      {isLoadingManual ? (
                        <div className="flex items-center gap-3 text-gray-500 py-8">
                          <Loader2 className="w-5 h-5 animate-spin text-cyan-400" />
                          <span className="font-light">Retrieving manual data...</span>
                        </div>
                      ) : manualError ? (
                        <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg p-4">
                          {manualError}
                        </div>
                      ) : manualData ? (
                        <div>
                          <p className="text-gray-300 leading-relaxed mb-6 whitespace-pre-wrap font-light text-sm">
                            {manualData.answer_text}
                          </p>
                          
                          {/* Component Images - Enhanced Display */}
                          {manualData.images && manualData.images.length > 0 && (
                            <div className="mb-6">
                              <h5 className="text-xs uppercase tracking-wider text-gray-500 mb-3">Component Diagrams</h5>
                              <div className={`grid ${manualData.images.length === 1 ? 'grid-cols-1' : 'grid-cols-2'} gap-3`}>
                                {manualData.images.map((img, i) => (
                                  <motion.div 
                                    key={i} 
                                    initial={{ opacity: 0, scale: 0.95 }}
                                    animate={{ opacity: 1, scale: 1 }}
                                    transition={{ delay: i * 0.1 }}
                                    className="relative rounded-lg overflow-hidden border border-cyan-500/30 bg-black/40 p-2 group cursor-pointer hover:border-cyan-500/60 transition-all"
                                    onClick={() => window.open(img, '_blank')}
                                  >
                                    <img 
                                      src={img} 
                                      alt={`${selectedHotspot.label} diagram ${i + 1}`} 
                                      className="w-full h-auto rounded group-hover:scale-105 transition-transform duration-300" 
                                    />
                                    <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-sm px-2 py-1 rounded text-xs text-cyan-400">
                                      Click to enlarge
                                    </div>
                                  </motion.div>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {/* Technical Tables */}
                          {manualData.tables && manualData.tables.length > 0 && (
                            <div className="space-y-3 mb-6">
                              <h5 className="text-xs uppercase tracking-wider text-gray-500 mb-3">Technical Specifications</h5>
                              {manualData.tables.map((table, i) => (
                                <div key={i} className="p-4 bg-black/60 rounded-lg text-xs font-mono text-gray-300 border border-white/10 overflow-x-auto">
                                  <pre className="whitespace-pre-wrap">{table}</pre>
                                </div>
                              ))}
                            </div>
                          )}
                        </div>
                      ) : (
                        <p className="text-gray-500 text-sm font-light">
                          Manual information will appear here
                        </p>
                      )}
                    </div>

                    <button
                      onClick={() => navigate(`/chat?mode=owner&component=${selectedHotspot.id}&brand=${brand}&model=${model}`)}
                      className="w-full mt-8 py-4 px-6 rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white font-light tracking-wide transition-all duration-300 flex items-center justify-between group"
                    >
                      <span>Ask About This Component</span>
                      <ChevronRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </button>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="relative bg-gradient-to-br from-gray-900/30 to-black/30 backdrop-blur-xl border border-white/5 rounded-2xl p-16 h-full flex items-center justify-center"
                >
                  <div className="text-center">
                    <div className="w-20 h-20 rounded-full bg-gradient-to-br from-cyan-500/10 to-blue-500/10 border border-cyan-500/20 flex items-center justify-center mx-auto mb-6">
                      <BookOpen className="w-10 h-10 text-cyan-400/50" />
                    </div>
                    <p className="text-gray-400 text-lg font-light mb-2">
                      Select a Component
                    </p>
                    <p className="text-gray-600 text-sm font-light">
                      Click any glowing point to view details
                    </p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

export default VehicleExplorer;