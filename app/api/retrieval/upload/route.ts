import { PDFLoader } from "langchain/document_loaders/fs/pdf";

import { NextRequest, NextResponse } from "next/server";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

import { createClient } from "@supabase/supabase-js";
import { SupabaseVectorStore } from "langchain/vectorstores/supabase";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";

// Before running, follow set-up instructions at
// https://js.langchain.com/docs/modules/indexes/vector_stores/integrations/supabase

/**
 * This handler takes input text, splits it into chunks, and embeds those chunks
 * into a vector store for later retrieval. See the following docs for more information:
 *
 * https://js.langchain.com/docs/modules/data_connection/document_transformers/text_splitters/recursive_text_splitter
 * https://js.langchain.com/docs/modules/data_connection/vectorstores/integrations/supabase
 */
export async function POST(req: NextRequest) {
  const form = await req.formData();
  const file = form.get("file") as File;

  // const fileArrayBuffer = await file.arrayBuffer();
  // const fileBuffer = Buffer.from(fileArrayBuffer);

  // Using langchain PDFLoader to extract the content of the document
  const docContent = await new PDFLoader(file, { splitPages: false })
    .load()
    .then((doc) => {
      return doc.map((page) => {
        return page.pageContent.replace(/\n/g, " "); // It is recommended to use the context string with no new lines
      });
    });

  // if (process.env.NEXT_PUBLIC_DEMO === "true") {
  //   return NextResponse.json(
  //     {
  //       error: [
  //         "Ingest is not supported in demo mode.",
  //         "Please set up your own version of the repo here: https://github.com/langchain-ai/langchain-nextjs-template",
  //       ].join("\n"),
  //     },
  //     { status: 403 },
  //   );
  // }

  try {
    const client = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_PRIVATE_KEY!,
    );

    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 256,
      chunkOverlap: 20,
    });

    const splitDocuments = await splitter.createDocuments(docContent);

    const vectorstore = await SupabaseVectorStore.fromDocuments(
      splitDocuments,
      new OpenAIEmbeddings(),
      {
        client,
        tableName: "documents",
        queryName: "match_documents",
      },
    );

    return NextResponse.json({ ok: true }, { status: 200 });
  } catch (e: any) {
    return NextResponse.json({ error: e.message }, { status: 500 });
  }
}
