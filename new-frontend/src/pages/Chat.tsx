// // import { useState, useEffect, useRef } from "react";
// // import { useNavigate, useSearchParams } from "react-router-dom";
// // import { ArrowLeft, Send, User, Wrench, Loader2 } from "lucide-react";
// // import { Button } from "@/components/ui/button";
// // import { Textarea } from "@/components/ui/textarea";
// // import { useToast } from "@/hooks/use-toast";

// // interface Message {
// //   role: "user" | "assistant";
// //   content: string;
// //   images?: string[];
// //   tables?: string[];
// // }

// // const Chat = () => {
// //   const navigate = useNavigate();
// //   const [searchParams] = useSearchParams();
// //   const mode = searchParams.get("mode") as "owner" | "mechanic" || "owner";
// //   const { toast } = useToast();
  
// //   const [messages, setMessages] = useState<Message[]>([]);
// //   const [input, setInput] = useState("");
// //   const [isLoading, setIsLoading] = useState(false);
// //   const messagesEndRef = useRef<HTMLDivElement>(null);

// //   const scrollToBottom = () => {
// //     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
// //   };

// //   useEffect(() => {
// //     scrollToBottom();
// //   }, [messages]);

// //   const handleSubmit = async (e: React.FormEvent) => {
// //     e.preventDefault();
// //     if (!input.trim() || isLoading) return;

// //     const userMessage: Message = { role: "user", content: input };
// //     setMessages(prev => [...prev, userMessage]);
// //     setInput("");
// //     setIsLoading(true);

// //     try {
// //       const response = await fetch("http://localhost:8000/api/ask", {
// //         method: "POST",
// //         headers: {
// //           "Content-Type": "application/json",
// //         },
// //         body: JSON.stringify({
// //           query: input,
// //           mode: mode,
// //         }),
// //       });

// //       if (!response.ok) {
// //         throw new Error("Failed to get response");
// //       }

// //       const data = await response.json();
// //       const assistantMessage: Message = {
// //         role: "assistant",
// //         content: data.answer.answer_text || "I apologize, but I couldn't generate a response.",
// //         images: data.answer.images || [],
// //         tables: data.answer.tables || [],
// //       };

// //       setMessages(prev => [...prev, assistantMessage]);
// //     } catch (error) {
// //       console.error("Error:", error);
// //       toast({
// //         title: "Error",
// //         description: "Failed to get response. Please make sure the backend server is running.",
// //         variant: "destructive",
// //       });
// //     } finally {
// //       setIsLoading(false);
// //     }
// //   };

// //   const modeConfig = {
// //     owner: {
// //       icon: User,
// //       color: "secondary",
// //       gradient: "bg-gradient-silver",
// //       glow: "shadow-glow-silver",
// //       name: "Vehicle Owner",
// //     },
// //     mechanic: {
// //       icon: Wrench,
// //       color: "accent-mechanic",
// //       gradient: "bg-gradient-gold",
// //       glow: "shadow-glow-gold",
// //       name: "Professional Mechanic",
// //     },
// //   };

// //   const config = modeConfig[mode];
// //   const ModeIcon = config.icon;

// //   return (
// //     <div className="min-h-screen bg-gradient-to-br from-background via-primary to-background flex flex-col">
// //       {/* Header */}
// //       <header className="border-b border-border bg-card/50 backdrop-blur-glass">
// //         <div className="container mx-auto px-4 py-4 flex items-center justify-between">
// //           <Button
// //             variant="ghost"
// //             size="sm"
// //             onClick={() => navigate("/")}
// //             className="gap-2"
// //           >
// //             <ArrowLeft className="w-4 h-4" />
// //             Change Mode
// //           </Button>
          
// //           <div className="flex items-center gap-3">
// //             <div className={`w-10 h-10 rounded-lg bg-${config.color}/10 flex items-center justify-center`}>
// //               <ModeIcon className={`w-5 h-5 text-${config.color}`} />
// //             </div>
// //             <div>
// //               <div className="text-sm font-medium">{config.name} Mode</div>
// //               <div className="text-xs text-muted-foreground">AI Assistant</div>
// //             </div>
// //           </div>

// //           <div className="w-24" /> {/* Spacer for centering */}
// //         </div>
// //       </header>

// //       {/* Chat Area */}
// //       <div className="flex-1 overflow-y-auto">
// //         <div className="container mx-auto px-4 py-8 max-w-4xl">
// //           {messages.length === 0 ? (
// //             <div className="text-center py-16 animate-fade-in">
// //               <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl ${config.gradient} flex items-center justify-center ${config.glow}`}>
// //                 <ModeIcon className="w-10 h-10 text-white" />
// //               </div>
// //               <h2 className="text-2xl font-bold mb-4">
// //                 Welcome to {config.name} Mode
// //               </h2>
// //               <p className="text-muted-foreground max-w-md mx-auto">
// //                 {mode === "owner" 
// //                   ? "Ask me anything about your vehicle. I'll provide clear, easy-to-understand answers."
// //                   : "Get detailed technical specifications and professional diagnostic information for your repairs."}
// //               </p>
// //             </div>
// //           ) : (
// //             <div className="space-y-6">
// //               {messages.map((message, index) => (
// //                 <div
// //                   key={index}
// //                   className={`flex gap-4 animate-fade-in-up ${
// //                     message.role === "user" ? "justify-end" : "justify-start"
// //                   }`}
// //                 >
// //                   {message.role === "assistant" && (
// //                     <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
// //                       <ModeIcon className="w-5 h-5 text-white" />
// //                     </div>
// //                   )}
                  
// //                   <div
// //                     className={`max-w-2xl rounded-2xl p-6 ${
// //                       message.role === "user"
// //                         ? "bg-muted"
// //                         : "bg-card/50 backdrop-blur-glass border border-border"
// //                     }`}
// //                   >
// //                     <p className="whitespace-pre-wrap leading-relaxed">{message.content}</p>
                    
// //                     {message.images && message.images.length > 0 && (
// //                       <div className="mt-4 grid grid-cols-2 gap-2">
// //                         {message.images.map((img, i) => (
// //                           <img
// //                             key={i}
// //                             src={img}
// //                             alt={`Diagram ${i + 1}`}
// //                             className="rounded-lg border border-border"
// //                           />
// //                         ))}
// //                       </div>
// //                     )}
                    
// //                     {message.tables && message.tables.length > 0 && (
// //                       <div className="mt-4 space-y-2">
// //                         {message.tables.map((table, i) => (
// //                           <div key={i} className="p-4 bg-background/50 rounded-lg text-sm font-mono">
// //                             {table}
// //                           </div>
// //                         ))}
// //                       </div>
// //                     )}
// //                   </div>

// //                   {message.role === "user" && (
// //                     <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
// //                       <User className="w-5 h-5" />
// //                     </div>
// //                   )}
// //                 </div>
// //               ))}
              
// //               {isLoading && (
// //                 <div className="flex gap-4 animate-fade-in">
// //                   <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
// //                     <ModeIcon className="w-5 h-5 text-white" />
// //                   </div>
// //                   <div className="max-w-2xl rounded-2xl p-6 bg-card/50 backdrop-blur-glass border border-border">
// //                     <div className="flex items-center gap-2">
// //                       <Loader2 className="w-4 h-4 animate-spin" />
// //                       <span className="text-muted-foreground">Analyzing your question...</span>
// //                     </div>
// //                   </div>
// //                 </div>
// //               )}
              
// //               <div ref={messagesEndRef} />
// //             </div>
// //           )}
// //         </div>
// //       </div>

// //       {/* Input Area */}
// //       <div className="border-t border-border bg-card/50 backdrop-blur-glass">
// //         <div className="container mx-auto px-4 py-6 max-w-4xl">
// //           <form onSubmit={handleSubmit} className="flex gap-4">
// //             <Textarea
// //               value={input}
// //               onChange={(e) => setInput(e.target.value)}
// //               placeholder={mode === "owner" 
// //                 ? "Ask about your vehicle..." 
// //                 : "Enter technical query..."}
// //               className="flex-1 min-h-[60px] max-h-[200px] bg-background/50 border-border focus:ring-2 focus:ring-accent resize-none"
// //               onKeyDown={(e) => {
// //                 if (e.key === "Enter" && !e.shiftKey) {
// //                   e.preventDefault();
// //                   handleSubmit(e);
// //                 }
// //               }}
// //             />
// //             <Button
// //               type="submit"
// //               disabled={isLoading || !input.trim()}
// //               className={`${config.gradient} hover:opacity-90 ${config.glow} h-[60px] px-8 text-background font-semibold`}
// //             >
// //               {isLoading ? (
// //                 <Loader2 className="w-5 h-5 animate-spin" />
// //               ) : (
// //                 <>
// //                   <Send className="w-5 h-5 mr-2" />
// //                   Ask
// //                 </>
// //               )}
// //             </Button>
// //           </form>
// //         </div>
// //       </div>
// //     </div>
// //   );
// // };

// // export default Chat;
// import { useState, useEffect, useRef } from "react";
// import { useNavigate, useSearchParams } from "react-router-dom";
// import { ArrowLeft, Send, User, Wrench, Loader2, Car } from "lucide-react";

// interface Message {
//   role: "user" | "assistant";
//   content: string;
//   images?: string[];
//   tables?: string[];
// }

// const Chat = () => {
//   const navigate = useNavigate();
//   const [searchParams] = useSearchParams();
//   const mode = searchParams.get("mode") as "owner" | "mechanic" || "owner";
//   const brand = searchParams.get("brand") || "";
//   const model = searchParams.get("model") || "";
  
//   const [messages, setMessages] = useState<Message[]>([]);
//   const [input, setInput] = useState("");
//   const [isLoading, setIsLoading] = useState(false);
//   const messagesEndRef = useRef<HTMLDivElement>(null);

//   const scrollToBottom = () => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   };

//   useEffect(() => {
//     scrollToBottom();
//   }, [messages]);

//   // Redirect if no car selected
//   useEffect(() => {
//     if (!brand || !model) {
//       navigate("/brand-selection");
//     }
//   }, [brand, model, navigate]);

//   const handleSubmit = async (e: React.FormEvent) => {
//     e.preventDefault();
//     if (!input.trim() || isLoading) return;

//     const userMessage: Message = { role: "user", content: input };
//     setMessages(prev => [...prev, userMessage]);
//     setInput("");
//     setIsLoading(true);

//     try {
//       const response = await fetch("http://localhost:8000/api/ask", {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           query: input,
//           manufacturer: brand,
//           model: model,
//           mode: mode,
//         }),
//       });

//       if (!response.ok) {
//         throw new Error("Failed to get response");
//       }

//       const data = await response.json();
//       const assistantMessage: Message = {
//         role: "assistant",
//         content: data.answer.answer_text || "I apologize, but I couldn't generate a response.",
//         images: data.answer.images || [],
//         tables: data.answer.tables || [],
//       };

//       setMessages(prev => [...prev, assistantMessage]);
//     } catch (error) {
//       console.error("Error:", error);
//       const errorMessage: Message = {
//         role: "assistant",
//         content: "Failed to get response. Please make sure the backend server is running.",
//       };
//       setMessages(prev => [...prev, errorMessage]);
//     } finally {
//       setIsLoading(false);
//     }
//   };

//   const modeConfig = {
//     owner: {
//       icon: User,
//       color: "secondary",
//       gradient: "bg-gradient-to-br from-blue-500 to-blue-600",
//       glow: "shadow-lg shadow-blue-500/50",
//       name: "Vehicle Owner",
//     },
//     mechanic: {
//       icon: Wrench,
//       color: "accent-mechanic",
//       gradient: "bg-gradient-to-br from-orange-500 to-orange-600",
//       glow: "shadow-lg shadow-orange-500/50",
//       name: "Professional Mechanic",
//     },
//   };

//   const config = modeConfig[mode];
//   const ModeIcon = config.icon;

//   // Format car name
//   const carName = `${brand.charAt(0).toUpperCase() + brand.slice(1)} ${model.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}`;

//   return (
//     <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex flex-col">
//       {/* Header */}
//       <header className="border-b border-slate-700 bg-slate-800/50 backdrop-blur-sm">
//         <div className="container mx-auto px-4 py-4 flex items-center justify-between">
//           <button
//             onClick={() => navigate(`/mode-selection?brand=${brand}&model=${model}`)}
//             className="flex items-center gap-2 px-4 py-2 rounded-lg hover:bg-slate-700/50 transition-colors"
//           >
//             <ArrowLeft className="w-4 h-4" />
//             Change Mode
//           </button>
          
//           <div className="flex items-center gap-4">
//             <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-slate-700/50">
//               <Car className="w-4 h-4 text-blue-400" />
//               <span className="text-sm font-medium">{carName}</span>
//             </div>
            
//             <div className={`flex items-center gap-3 px-4 py-2 rounded-lg ${config.gradient}`}>
//               <ModeIcon className="w-5 h-5 text-white" />
//               <div>
//                 <div className="text-sm font-medium text-white">{config.name}</div>
//               </div>
//             </div>
//           </div>

//           <div className="w-32" />
//         </div>
//       </header>

//       {/* Chat Area */}
//       <div className="flex-1 overflow-y-auto">
//         <div className="container mx-auto px-4 py-8 max-w-4xl">
//           {messages.length === 0 ? (
//             <div className="text-center py-16 animate-fade-in">
//               <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl ${config.gradient} flex items-center justify-center ${config.glow}`}>
//                 <ModeIcon className="w-10 h-10 text-white" />
//               </div>
//               <h2 className="text-2xl font-bold mb-4 text-white">
//                 Welcome to {config.name} Mode
//               </h2>
//               <p className="text-slate-400 max-w-md mx-auto mb-4">
//                 {mode === "owner" 
//                   ? "Ask me anything about your vehicle. I'll provide clear, easy-to-understand answers."
//                   : "Get detailed technical specifications and professional diagnostic information for your repairs."}
//               </p>
//               <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-slate-700/50 text-sm text-slate-300">
//                 <Car className="w-4 h-4" />
//                 Searching in: {carName} Manual
//               </div>
//             </div>
//           ) : (
//             <div className="space-y-6">
//               {messages.map((message, index) => (
//                 <div
//                   key={index}
//                   className={`flex gap-4 ${
//                     message.role === "user" ? "justify-end" : "justify-start"
//                   }`}
//                 >
//                   {message.role === "assistant" && (
//                     <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
//                       <ModeIcon className="w-5 h-5 text-white" />
//                     </div>
//                   )}
                  
//                   <div
//                     className={`max-w-2xl rounded-2xl p-6 ${
//                       message.role === "user"
//                         ? "bg-slate-700"
//                         : "bg-slate-800 border border-slate-700"
//                     }`}
//                   >
//                     <p className="whitespace-pre-wrap leading-relaxed text-slate-100">{message.content}</p>
                    
//                     {message.images && message.images.length > 0 && (
//                       <div className="mt-4 grid grid-cols-2 gap-2">
//                         {message.images.map((img, i) => (
//                           <img
//                             key={i}
//                             src={img}
//                             alt={`Diagram ${i + 1}`}
//                             className="rounded-lg border border-slate-600"
//                           />
//                         ))}
//                       </div>
//                     )}
                    
//                     {message.tables && message.tables.length > 0 && (
//                       <div className="mt-4 space-y-2">
//                         {message.tables.map((table, i) => (
//                           <div key={i} className="p-4 bg-slate-900/50 rounded-lg text-sm font-mono text-slate-300">
//                             {table}
//                           </div>
//                         ))}
//                       </div>
//                     )}
//                   </div>

//                   {message.role === "user" && (
//                     <div className="w-10 h-10 rounded-lg bg-slate-700 flex items-center justify-center flex-shrink-0">
//                       <User className="w-5 h-5 text-slate-300" />
//                     </div>
//                   )}
//                 </div>
//               ))}
              
//               {isLoading && (
//                 <div className="flex gap-4">
//                   <div className={`w-10 h-10 rounded-lg ${config.gradient} flex items-center justify-center flex-shrink-0`}>
//                     <ModeIcon className="w-5 h-5 text-white" />
//                   </div>
//                   <div className="max-w-2xl rounded-2xl p-6 bg-slate-800 border border-slate-700">
//                     <div className="flex items-center gap-2">
//                       <Loader2 className="w-4 h-4 animate-spin text-blue-400" />
//                       <span className="text-slate-400">Searching {carName} manual...</span>
//                     </div>
//                   </div>
//                 </div>
//               )}
              
//               <div ref={messagesEndRef} />
//             </div>
//           )}
//         </div>
//       </div>

//       {/* Input Area */}
//       <div className="border-t border-slate-700 bg-slate-800/50 backdrop-blur-sm">
//         <div className="container mx-auto px-4 py-6 max-w-4xl">
//           <form onSubmit={handleSubmit} className="flex gap-4">
//             <textarea
//               value={input}
//               onChange={(e) => setInput(e.target.value)}
//               placeholder={mode === "owner" 
//                 ? "Ask about your vehicle..." 
//                 : "Enter technical query..."}
//               className="flex-1 min-h-[60px] max-h-[200px] bg-slate-900 border border-slate-700 rounded-lg px-4 py-3 text-slate-100 placeholder-slate-500 focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
//               onKeyDown={(e) => {
//                 if (e.key === "Enter" && !e.shiftKey) {
//                   e.preventDefault();
//                   handleSubmit(e);
//                 }
//               }}
//             />
//             <button
//               type="submit"
//               disabled={isLoading || !input.trim()}
//               className={`${config.gradient} hover:opacity-90 ${config.glow} h-[60px] px-8 rounded-lg text-white font-semibold disabled:opacity-50 disabled:cursor-not-allowed transition-all`}
//             >
//               {isLoading ? (
//                 <Loader2 className="w-5 h-5 animate-spin" />
//               ) : (
//                 <div className="flex items-center gap-2">
//                   <Send className="w-5 h-5" />
//                   Ask
//                 </div>
//               )}
//             </button>
//           </form>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default Chat;
import { useState, useEffect, useRef } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { ArrowLeft, Send, User, Wrench, Loader2, Car, BookOpen, MessageSquare, Zap, FileText } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Message {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  tables?: string[];
}

const Chat = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const mode = searchParams.get("mode") as "owner" | "mechanic" || "owner";
  const brand = searchParams.get("brand") || "";
  const model = searchParams.get("model") || "";
  const component = searchParams.get("component") || "";
  
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (!brand || !model) {
      navigate("/brand-selection");
    }
  }, [brand, model, navigate]);

  // Enhanced prompt based on mode and component
  const enhanceQuery = (userQuery: string) => {
    let enhancedQuery = userQuery;
    
    if (mode === "mechanic") {
      enhancedQuery = `As a professional mechanic, provide detailed technical information with step-by-step procedures, specifications, and diagrams where applicable. Query: ${userQuery}`;
    } else if (component) {
      enhancedQuery = `Regarding the ${component} component, provide clear and easy-to-understand information with visual aids if available. Query: ${userQuery}`;
    } else {
      enhancedQuery = `Provide a clear, easy-to-understand answer suitable for a vehicle owner with visual aids where helpful. Query: ${userQuery}`;
    }
    
    return enhancedQuery;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const enhancedQuery = enhanceQuery(input);
      
      const response = await fetch("http://localhost:8000/api/ask", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: enhancedQuery,
          manufacturer: brand,
          model: model,
          mode: mode,
        }),
      });

      if (!response.ok) {
        throw new Error("Failed to get response");
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: "assistant",
        content: data.answer.answer_text || "I apologize, but I couldn't generate a response.",
        images: data.answer.images || [],
        tables: data.answer.tables || [],
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Error:", error);
      const errorMessage: Message = {
        role: "assistant",
        content: "Failed to get response. Please make sure the backend server is running.",
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const modeConfig = {
    owner: {
      icon: User,
      color: "blue",
      gradient: "from-blue-600 to-cyan-600",
      name: "Owner Mode",
      description: "Clear answers with visual guides",
      bgPattern: "from-blue-500/5 to-cyan-500/5"
    },
    mechanic: {
      icon: Wrench,
      color: "orange",
      gradient: "from-orange-600 to-red-600",
      name: "Mechanic Mode",
      description: "Technical specs & step-by-step procedures",
      bgPattern: "from-orange-500/5 to-red-500/5"
    },
  };

  const config = modeConfig[mode];
  const ModeIcon = config.icon;

  const carName = `${brand.charAt(0).toUpperCase() + brand.slice(1)} ${model.split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')}`;

  const quickActions = mode === "mechanic" 
    ? [
        { icon: Wrench, label: "Diagnostic Procedures", query: "Show me the diagnostic procedures" },
        { icon: FileText, label: "Technical Specs", query: "What are the technical specifications?" },
        { icon: Zap, label: "Troubleshooting Guide", query: "Common problems and solutions" }
      ]
    : [
        { icon: BookOpen, label: "Maintenance Schedule", query: "What's the maintenance schedule?" },
        { icon: MessageSquare, label: "Common Questions", query: "What are common questions about this vehicle?" },
        { icon: Zap, label: "Quick Tips", query: "Give me quick tips for this vehicle" }
      ];

  return (
    <div className="min-h-screen bg-black text-white relative overflow-hidden">
      {/* Premium Background */}
      <div className="fixed inset-0 z-0">
        <div className="absolute inset-0 bg-gradient-to-br from-black via-gray-900 to-black" />
        <div className={`absolute inset-0 bg-gradient-to-br ${config.bgPattern} opacity-30`} />
        
        {/* Animated grid */}
        <div className="absolute inset-0 opacity-10">
          <div className="absolute inset-0" style={{
            backgroundImage: 'linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)',
            backgroundSize: '50px 50px'
          }} />
        </div>
      </div>

      {/* Header */}
      <header className="relative z-50 border-b border-white/10 bg-black/40 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <button
              onClick={() => navigate(mode === "mechanic" 
                ? `/dashboard/mechanic?brand=${brand}&model=${model}` 
                : `/dashboard/owner?brand=${brand}&model=${model}`
              )}
              className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors group"
            >
              <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
              <span className="text-sm tracking-wider uppercase">Back to Dashboard</span>
            </button>
            
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10">
                <Car className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-light tracking-wide">{carName}</span>
              </div>
              
              <div className={`flex items-center gap-3 px-4 py-2 rounded-full bg-gradient-to-r ${config.gradient}`}>
                <ModeIcon className="w-4 h-4 text-white" />
                <span className="text-sm font-medium text-white">{config.name}</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <div className="relative z-10 max-w-7xl mx-auto px-6 py-8 grid lg:grid-cols-3 gap-6 h-[calc(100vh-180px)]">
        
        {/* Left Sidebar - Context & Quick Actions */}
        <div className="lg:col-span-1 space-y-4 overflow-y-auto">
          {/* Context Card */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-gradient-to-br from-gray-900/80 to-black/80 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${config.gradient} flex items-center justify-center`}>
                <ModeIcon className="w-6 h-6 text-white" />
              </div>
              <div>
                <h3 className="font-semibold">{config.name}</h3>
                <p className="text-xs text-gray-400">{config.description}</p>
              </div>
            </div>
            
            {component && (
              <div className="mt-4 p-3 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
                <p className="text-xs text-cyan-400 uppercase tracking-wider mb-1">Focused On</p>
                <p className="text-sm font-medium">{component}</p>
              </div>
            )}
          </motion.div>

          {/* Quick Actions */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-gradient-to-br from-gray-900/80 to-black/80 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
          >
            <h3 className="text-sm uppercase tracking-wider text-gray-400 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              {quickActions.map((action, idx) => (
                <button
                  key={idx}
                  onClick={() => setInput(action.query)}
                  className="w-full flex items-center gap-3 p-3 rounded-lg bg-white/5 hover:bg-white/10 border border-white/5 hover:border-white/20 transition-all text-left group"
                >
                  <action.icon className="w-4 h-4 text-gray-400 group-hover:text-white transition-colors" />
                  <span className="text-sm text-gray-300 group-hover:text-white transition-colors">{action.label}</span>
                </button>
              ))}
            </div>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="bg-gradient-to-br from-gray-900/80 to-black/80 backdrop-blur-xl border border-white/10 rounded-2xl p-6"
          >
            <h3 className="text-sm uppercase tracking-wider text-gray-400 mb-4">Session Info</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Questions Asked</span>
                <span className="text-sm font-semibold text-white">{Math.floor(messages.length / 2)}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-xs text-gray-400">Mode</span>
                <span className={`text-sm font-semibold bg-gradient-to-r ${config.gradient} bg-clip-text text-transparent`}>
                  {mode === "mechanic" ? "Professional" : "Standard"}
                </span>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Main Chat Area */}
        <div className="lg:col-span-2 flex flex-col bg-gradient-to-br from-gray-900/50 to-black/50 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden">
          {/* Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.length === 0 ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center max-w-md">
                  <div className={`w-20 h-20 mx-auto mb-6 rounded-2xl bg-gradient-to-br ${config.gradient} flex items-center justify-center`}>
                    <ModeIcon className="w-10 h-10 text-white" />
                  </div>
                  <h2 className="text-2xl font-bold mb-3">
                    {mode === "mechanic" ? "Professional Technical Assistant" : "Your Vehicle Assistant"}
                  </h2>
                  <p className="text-gray-400 text-sm mb-6">
                    {mode === "mechanic" 
                      ? "Get detailed technical specifications, diagnostic procedures, and repair instructions with diagrams and tables."
                      : "Ask anything about your vehicle. Get clear answers with helpful visual guides."}
                  </p>
                  <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 border border-white/10 text-xs text-gray-400">
                    Analyzing: {carName} Manual
                  </div>
                </div>
              </div>
            ) : (
              <>
                {messages.map((message, index) => (
                  <motion.div
                    key={index}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className={`flex gap-4 ${message.role === "user" ? "justify-end" : "justify-start"}`}
                  >
                    {message.role === "assistant" && (
                      <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${config.gradient} flex items-center justify-center flex-shrink-0`}>
                        <ModeIcon className="w-5 h-5 text-white" />
                      </div>
                    )}
                    
                    <div
                      className={`max-w-2xl rounded-2xl p-6 ${
                        message.role === "user"
                          ? "bg-white/10 border border-white/20"
                          : "bg-black/40 border border-white/10"
                      }`}
                    >
                      <p className="whitespace-pre-wrap leading-relaxed text-gray-100 text-sm">{message.content}</p>
                      
                      {message.images && message.images.length > 0 && (
                        <div className="mt-4 grid grid-cols-2 gap-3">
                          {message.images.map((img, i) => (
                            <div key={i} className="rounded-lg overflow-hidden border border-white/20">
                              <img src={img} alt={`Diagram ${i + 1}`} className="w-full" />
                            </div>
                          ))}
                        </div>
                      )}
                      
                      {message.tables && message.tables.length > 0 && (
                        <div className="mt-4 space-y-3">
                          {message.tables.map((table, i) => (
                            <div key={i} className="p-4 bg-black/60 rounded-lg text-xs font-mono text-gray-300 border border-white/10 overflow-x-auto">
                              {table}
                            </div>
                          ))}
                        </div>
                      )}
                    </div>

                    {message.role === "user" && (
                      <div className="w-10 h-10 rounded-lg bg-white/10 border border-white/20 flex items-center justify-center flex-shrink-0">
                        <User className="w-5 h-5 text-gray-300" />
                      </div>
                    )}
                  </motion.div>
                ))}
                
                {isLoading && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex gap-4"
                  >
                    <div className={`w-10 h-10 rounded-lg bg-gradient-to-br ${config.gradient} flex items-center justify-center flex-shrink-0`}>
                      <ModeIcon className="w-5 h-5 text-white" />
                    </div>
                    <div className="max-w-2xl rounded-2xl p-6 bg-black/40 border border-white/10">
                      <div className="flex items-center gap-3">
                        <Loader2 className="w-4 h-4 animate-spin text-cyan-400" />
                        <span className="text-gray-400 text-sm">
                          {mode === "mechanic" 
                            ? "Analyzing technical specifications..." 
                            : "Searching manual..."}
                        </span>
                      </div>
                    </div>
                  </motion.div>
                )}
                
                <div ref={messagesEndRef} />
              </>
            )}
          </div>

          {/* Input Area */}
          <div className="border-t border-white/10 p-6">
            <form onSubmit={handleSubmit} className="flex gap-3">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder={mode === "mechanic" 
                  ? "Enter technical query..." 
                  : "Ask about your vehicle..."}
                className="flex-1 min-h-[60px] max-h-[120px] bg-black/40 border border-white/20 rounded-xl px-4 py-3 text-gray-100 placeholder-gray-500 focus:ring-2 focus:ring-cyan-500/50 focus:border-transparent resize-none text-sm"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSubmit(e);
                  }
                }}
              />
              <button
                type="submit"
                disabled={isLoading || !input.trim()}
                className={`bg-gradient-to-r ${config.gradient} hover:opacity-90 h-[60px] px-6 rounded-xl text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center gap-2`}
              >
                {isLoading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    <span className="hidden sm:inline">Send</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chat;